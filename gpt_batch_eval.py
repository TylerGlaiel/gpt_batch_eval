import random
import openai
import backoff
import string
import copy
from dataclasses import dataclass

@dataclass
class _gpt_batch_entry:
    data: str
    index: int
    results: list[str]
    retries: int
    
    
class gpt_batch_eval:
    api_key = None
    
    def __init__(self, 
        #parameters specific to gptbatch
        command, #the english-language "function" that should be evaluated over the data provided
        data, #a list of strings that the batch should process
        batch_size = 25, #the size of each batch sent to openAI (data is grouped into batches to save on requests and tokens, experiment to find the ideal batch size for your specific request, larger is better if you're trying to save on API calls)
        shuffle_data = True, #if true, data batches will be selected randomly from the provided data. If false, data batches will be sent in order. Sending in order may cause GPT (well, gpt-3.5-turbo at least) to get stuck in a pattern and return the same result for every entry in a batch, randomizing mitigates the chances of this happening
        multiplicity = 1, #the number of times each data entry is run through ChatGPT. If more than 1, results returned will include a list of all results returned for that piece of data. If results are sometimes innaccurate, set multiplicity above 1 and then do what you need to do with the "multiple opinions" returned. This significantly increases requests and tokens used.
        max_retry = 3, #if a piece of data fails to validate or return a result, it will be shuffled back into the queue to try again in a new batch, up to this many times (to prevent one bad piece of data from getting looped repeatedly)
        validate = None, #a callback of the form validate(data, result). Return True if the result looks good, and False if it needs to retry that data in a new batch
        max_requests = -1, #limits the number of calls to the OpenAI API. If the limit is hit, you will only get partial results from run()
        max_tokens = -1, #limits the number of tokens the OpenAI API uses. If the limit is hit, you will only get partial results from run(). This is not a strict limit, the final request can put you over the max here (then it will stop)
        
        logfile = "gpt_batch_log.txt", #every time a batch is processed, outputs the list of entries in the batch and the response from ChatGPT into the log file. If your program crashes mid-batch you can try to recover from this file so as not to need to rerun the entire batch again (todo)
        
        #how to format the command and data before getting sent to chatGPT. unlikely to need to change this
        batch_format = "Evaluate the following command against the provided data. Format results as <index>: <result>, one per line. Do not repeat the data. Do not elaborate on the results. \ncommand: {command}\ndata:\n",
        data_format = "{index}: {data}\n",

        #parameters forwarded to GPT Api
        model = "best_available", #"best_available" means gpt-4 if you have access to gpt-4, otherwise gpt-3.5-turbo
        temperature = 0, #temperature is 0 because we dont want gpt to be creative in its responses, we want it to be a COLD, UNCARING ROBOT. but you can turn this up if you do want more "creative" results
    ):
        self.command = command
        self.data = data
        self.batch_size = batch_size
        self.shuffle_data = shuffle_data
        self.multiplicity = multiplicity
        self.max_retry = max_retry
        self.validate = validate
        self.max_requests = max_requests
        self.max_tokens = max_tokens
        self.batch_format = batch_format
        self.data_format = data_format
        self.logfile = logfile
        self.request_count = 0
        self.token_count = 0
        self.model = model
        self.temperature = temperature
        self.warnings = False
        
        if multiplicity > 1 and temperature == 0 and shuffle_data == False:
            print("WARNING: Multiplicity is set to more than 1, you need to also set shuffle_data to True or set temperature to a high value for this to have an effect")
            self.warnings = True
            
        if len(data) != len(set(data)):
            print("WARNING: data set has duplicate entries.")
            self.warnings = True
            
        if max_requests > 0 and (len(data) * self.multiplicity) / batch_size > max_requests:
            print("WARNING: max_requests is too small to process the amount of data you provided with the given batch_size.")
            self.warnings = True
     
    #Processes the batch. If there were warnings when creating the batch, you must pass True here to ignore them, or fix the warnings and try again, otherwise the batch won't run (light protection against wasting API requests)
    def run(self, ignore_warnings = False):
        if(gpt_batch_eval.api_key == None and openai.api_key == None):
            raise RuntimeError("Please set your open API key with gpt_batch_eval.api_key = \"my api key\" or openai.api_key = \"my api key\" before calling run")
            
        if self.warnings and not ignore_warnings:
            raise RuntimeError("Please fix warnings before calling run, or pass in ignore_warnings=True to run()")
            
        self.verbose_log = open(self.logfile, "w")
        processed_count = 0
        
        self.__prepare_run()
        total_data = len(self.data_queue)

        while self.__any_batches_left() and not self.__at_request_limit():
            try:
                processed_count += self.__run_next_batch()
            except Exception as e:
                print(e)
                
            print("data processed: "+str(processed_count)+"/"+str(total_data))
            
        if self.__any_batches_left():
            print("request or token limit hit, returning partial results")
            
        return self.__flattened_results()
    
    #not implemented yet, but the idea here is to load the verbose_log or a pickle from a previous incomplete run and pick up where it left off, instead of needing to run the whole batch again
    def continue_run(self):
        raise NotImplementedError("continue run is not implemented yet")
    
    #does not use API requests, simply generates one batch and returns the prompt the batch would use so you can review that it looks correct
    def preview(self):
        self.__prepare_run()
        
        return self.__construct_prompt(self.__next_batch())
        
    #like "run" except only runs ONE batch so you can make sure the results look good for your command before running the whole thing
    def test(self):
        self.__prepare_run()
        
        batch_results = []
        
        if self.__any_batches_left() and not self.__at_request_limit():
            self.__run_next_batch()
        
        return self.__flattened_results()
    
    #private functions
    @backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.ServiceUnavailableError))
    def __chat_completions_with_backoff(self, **kwargs):
        return openai.ChatCompletion.create(**kwargs)
        
    def __construct_prompt(self, databatch):
        res = self.batch_format.format(command=self.command)
        for i in range(0, len(databatch)):
            res += self.data_format.format(index=i, data=databatch[i])
        return res

    def __run_prompt(self, prompt):
        if(gpt_batch_eval.api_key): openai.api_key = gpt_batch_eval.api_key
        if self.model == "best_available" and self.__has_access_to_gpt4():
            self.model = "gpt-4"
        else:
            self.model = "gpt-3.5-turbo"
        
        self.request_count += 1
        
        res = self.__chat_completions_with_backoff(
            model = self.model, 
            temperature = self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        
        self.token_count += res['usage']['total_tokens']
        
        return res['choices'][0]['message']['content']

    def __prepare_run(self):
        self.verbose_log_file = open(self.logfile, "w", encoding="utf-8")
        self.data_queue = copy.copy(self.data) * self.multiplicity
        
        self.results = {}
        for i in range(0, len(self.data)):
            self.results[self.data[i]] = _gpt_batch_entry(data = self.data[i], index = i, results = [], retries = 0)
            
    def __next_batch(self):
        batch = []
    
        while len(batch) < self.batch_size and len(self.data_queue) > 0:
            index = 0
            if self.shuffle_data:
                index = random.randrange(len(self.data_queue))
            batch.append(self.data_queue.pop(index))
        
        return batch
        
    def __any_batches_left(self):
        return len(self.data_queue) > 0
    
    def __at_request_limit(self):
        if self.max_requests > 0 and self.request_count >= self.max_requests:
            return True
            
        if self.max_tokens > 0 and self.token_count >= self.max_tokens:
            return True
            
        return False
        
    def __run_next_batch(self):
        batch = self.__next_batch()

        try:
            if self.verbose_log_file: 
                print("names: "+str(batch), file=self.verbose_log_file)
            prompt = self.__construct_prompt(batch)
            output = self.__run_prompt(prompt)
            if self.verbose_log_file: 
                print(output, file=self.verbose_log_file)
                self.verbose_log_file.flush()
            successful_results, failed_results = self.__decode_response(batch, output)
            complete_count = len(successful_results)
        except:
            failed_results = batch
            successful_results = []
            complete_count = 0
            
        for data, result in successful_results:
            self.results[data].results.append(result)
        
            
        for failure in failed_results:
            if self.results[failure].retries < self.max_retry:
                self.results[failure].retries += 1
                if len(self.data_queue) == 0:
                    self.data_queue.append(failure)
                else:
                    self.data_queue.insert(random.randrange(len(self.data_queue)), failure)
            else: #we aren't going to retry this one anymore, so count it as "complete" for progress bar reasons
                complete_count += 1
                
        return complete_count
        
    def __decode_response(self, batch, gptresponse):
        res = []
        failed = []
        succeeded = []
        decoded = self.__split_response(gptresponse)
        for index, result in decoded:
            if index >= 0 and index < len(batch):
                data = batch[index]

                if self.validate is None or self.validate(data, result):
                    res.append((data, result))
                    succeeded.append(data)
        
        for data in batch:
            if not data in succeeded:
                failed.append(data)
        
        return res, failed
        
        
    def __split_response(self, gptresponse):
        result = []
        indices = set()
        lines = gptresponse.strip().split('\n')
        
        for line in lines:
            try:
                index, _, text = line.partition(':')
                index = int(index)
                if index in indices:
                    print("batch contained duplicate indices in result, whole batch considered malformed")
                    return []
                indices.add(index)
                result.append((index, text.strip()))
            except ValueError:
                continue
        
        return result
        
    def __flattened_results(self):
        flat_results = sorted(list(self.results.values()), key = lambda result : result.index)
        
        res = []
        for result in flat_results:
            output = ""
            if len(result.results) == 1: 
                output = result.results[0]
            elif len(result.results) > 1:
                output = result.results
            
            res.append((result.data, output)) 
        
        return res
        
    def __has_access_to_gpt4(self):
        if(gpt_batch_eval.api_key): openai.api_key = gpt_batch_eval.api_key
        
        models = openai.Model.list()

        for i in models['data']:
            if i['id'] == "gpt-4":
                return True
        
        return False