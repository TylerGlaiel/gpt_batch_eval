# gpt_batch_eval
Runs an "english language function" over a set of data via ChatGPT. Data is grouped into batches to reduce API calls

USAGE:

```
from gpt_batch_eval import gpt_batch_eval
gpt_batch_eval.api_key = "put your openai api key here"

data = ["knife", "fork", "gun", "sword", "spoon", "spork", "spife", "chopstick", "throwing spoon"]
batch = gpt_batch_eval("classify as cutlery or weapon", data)
print(batch.run())
#prints [('knife', 'cutlery'), ('fork', 'cutlery'), ('gun', 'weapon'), ('sword', 'weapon'), ('spoon', 'cutlery'), ('spork', 'cutlery'), ('spife', 'cutlery'), ('chopstick', 'cutlery'), ('throwing spoon', 'cutlery')]
```

Data gets chunked into batches of size at most batch_size (default 25), each batch uses 1 API call. 

If you want to review your command to make sure it looks good before running the entire thing, you can call "batch.preview()" to preview what the ChatGPT prompt will look like for a single batch, or "batch.test()" to run exactly one batch through chatGPT and see what the results looks like

documentation for parameters for gpt_batch_eval(): (copied from the source code)

```
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
```
