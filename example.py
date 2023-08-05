from gpt_batch_eval import gpt_batch_eval
gpt_batch_eval.api_key = "put your openai api key here" #alternatively, import openai and set it there instead

def ValidResult(data, result): #example of a validation function
    if result == "use" or result == "":
        return False
    else:
        return True

data = ["knife", "fork", "gun", "sword", "spoon", "spork", "spife", "chopstick", "throwing spoon"]

batch1 = gpt_batch_eval("output the verb that describes how you use this item", data, validate=ValidResult)
print(batch1.run())
#prints [('knife', 'cut'), ('fork', 'eat'), ('gun', 'shoot'), ('sword', 'fight'), ('spoon', 'eat'), ('spork', 'eat'), ('spife', 'eat'), ('chopstick', 'eat'), ('throwing spoon', 'throw')]

batch2 = gpt_batch_eval("classify as cutlery or weapon", data)
print(batch2.run())
#prints [('knife', 'cutlery'), ('fork', 'cutlery'), ('gun', 'weapon'), ('sword', 'weapon'), ('spoon', 'cutlery'), ('spork', 'cutlery'), ('spife', 'cutlery'), ('chopstick', 'cutlery'), ('throwing spoon', 'cutlery')]

batch3 = gpt_batch_eval("translate into spanish", data)
print(batch3.run())
#prints [('knife', 'cuchillo'), ('fork', 'tenedor'), ('gun', 'pistola'), ('sword', 'espada'), ('spoon', 'cuchara'), ('spork', 'cuchara tenedor'), ('spife', 'cuchillo y tenedor'), ('chopstick', 'palillo'), ('throwing spoon', 'cuchara de lanzamiento')]
