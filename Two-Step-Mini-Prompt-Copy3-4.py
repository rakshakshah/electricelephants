#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import re
import time
import sys

words = ["Optimistic",
         "Euphoric",
         "Liberating",
         "Heartwarming",
         "Romantic",
         "Seductive",
         "Triumphant",
         "Peaceful",
         "Inspiring",
         "Depressing",
         "Heartbreaking",
         "Defeated",
         "Somber",
         "Bittersweet",
         "Angry",
         "Emotional",
         "Tense",
         "Mysterious",
         "Lighthearted",
         "Fun",
         "Lonely",
         "Disturbing",
         "Scary",
         "Thrilling",
         "Dramatic",
         "Sincere",
         "Funny",
         "Frenzied",
         "Boyish",
         "Mature"]


def initialize_chain():
    """Initialize or reinitialize the conversation chain with Ollama."""
    global conversation_chain, conversation_memory
    conversation_memory = ConversationBufferMemory()
    conversation_chain = ConversationChain(
    llm=ollama,
    memory=conversation_memory,
    verbose=False)
    return conversation_chain

global words_str
words_str = ""
for word in words:
    words_str = words_str + ", " + word

words_str = words_str[2:]
words_str = words_str.upper()
words_str
#print(len(words))
for value in words:
    count = 0
    for i in words:
        if (i == value):
            count = count + 1
    #print(value, count)


# In[2]:


## prompts

## analyze the movie
def prompt1():
    return f"""Based on the character development, music soundtrack, and major plot points, how would you describe the sentiment 
of the movie {movie}? What emotions dominate the movie? How do viewers feel after watching? Here are some words to consider in your output:
{words_str}
IMPORTANT: RESPOND IN 350 WORDS OR LESS."""

## choose one side
def prompt2(word, memory):
    return f"""{memory}
Based on this analysis, is this movie more {word} than the average movie? Answer in 1 word ONLY using yes or no."""

def prompt3(word, memory):
    return f"""{memory}


YOUR GOAL: 
Based on the analysis above, on a scale from 0.0001 to 0.9999, how {word} is this movie? 
IMPORTANT: Respond with ONLY ONE numerical value with 4 decimal points. DO NOT INCLUDE ANY OTHER WORDS OR COMMENTARY."""

## shortened prompts

def prompt1short():
    return f"""Based on the character development, music soundtrack, and major plot points, how would you describe the sentiment 
of the movie {movie}? What emotions dominate the movie? How do viewers feel after watching?"""

def prompt2short(word):
    return f"""Based on this analysis, is this movie more {word} than the average movie? Answer in 1 word ONLY using yes or no."""

def prompt3short(word):
    return f"""Based on the analysis above, on a scale from 0.0001 to 0.9999, how {word} is this movie? 
IMPORTANT: Respond with ONLY ONE numerical value with 4 decimal points."""


# In[3]:


## define analyze_question
def analyze_question():
    # Initialize the chain if not already done
    global conversation_chain
    if conversation_chain is None:
        initialize_chain()
        
    comprehensive_prompt = prompt1()
    
    # Get response using the conversation chain, which maintains history
    model_start_time = time.time()

    #print(word)
    
    print("\nRAW MODEL RESPONSE:")
    
    # Replace the single predict call with a streaming approach
    full_response = ""
    
    for token in conversation_chain.llm.stream(comprehensive_prompt):
        #print(token, end="", flush=True)  # Print each token as it arrives
        full_response += token
    
    print()  # Add a newline after streaming completes
    
    model_end_time = time.time()
    #print(f"Analysis complete (Model processing took {model_end_time - model_start_time:.2f} seconds)")

    conversation_memory.save_context({"input": prompt1short()}, {"output": full_response})
    
    # Store the full_response for parsing
    response = full_response

    return model_start_time - model_end_time, response


# In[14]:


## define pick_word

def pick_word(word):

    memory = conversation_memory.load_memory_variables({}).get("history", "")
    
    comprehensive_prompt = prompt2(word, memory)
    model_start_time = time.time()
    
    #print("\nRAW MODEL RESPONSE:")
    
    # Replace the single predict call with a streaming approach
    full_response = ""
    
    for token in conversation_chain.llm.stream(comprehensive_prompt):
        #print(token, end="", flush=True)  # Print each token as it arrives
        full_response += token
    
    #print()  # Add a newline after streaming completes
    
    model_end_time = time.time()
    print(f"Analysis complete (Model processing took {model_end_time - model_start_time:.2f} seconds)")

    shortened_prompt2 = prompt2short(word)

    conversation_memory.save_context({"input": shortened_prompt2}, {"output": full_response})

    return model_start_time - model_end_time, full_response


# In[18]:


## define give_score
def give_score(word, memory):

    #memory = conversation_memory.load_memory_variables({}).get("history", "")
    
    comprehensive_prompt = prompt3(word, memory)
    model_start_time = time.time()
    
    # Replace the single predict call with a streaming approach
    full_response = ""
    
    for token in conversation_chain.llm.stream(comprehensive_prompt):
        #print(token, end="", flush=True)  # Print each token as it arrives
        full_response += token
    
    #print()  # Add a newline after streaming completes
    
    model_end_time = time.time()
    print(f"Analysis complete (Model processing took {model_end_time - model_start_time:.2f} seconds)")

    print(full_response)

    global scoretimes
    scoretimes.append(model_end_time - model_start_time)

    shortened_prompt = prompt3short(word)

    #print(shortened_prompt)

    conversation_memory.save_context({"input": shortened_prompt}, {"output": full_response})

    return model_start_time - model_end_time, full_response


# In[17]:


def home(gpu):
## trying different models
    global ollama
    # Global memory that will persist between function calls
    ollama = Ollama(
        model="llama3.2:latest",
        num_gpu = gpu,
        num_ctx = 2048
    )
    conversation_memory = ConversationBufferMemory()
    conversation_chain = None
    
    ### PASS THE MOVIE STRING HERE ###
    global movie
    title = "The Matrix"
    director = ["Lana Wachowski", "Lily Wachowski"]
    year = "1999"
    movie = title + " (" + year + ")" + " directed by " + director[0]
    print(movie)
    
    global scoretimes
    scoretimes = []
        
    initialize_chain()
    
    # Initialize the conversation chain
    
    all_words = words
    
    words_list = []
    score_list = []
    
    fulltime1 = time.time()
    
    prompt_time, movieanalysis = analyze_question()
    print(conversation_memory.load_memory_variables({}).get("history"))

    essay_time = time.time() - fulltime1
    
    print("TOTAL ESSAY TIME:", essay_time)
    
    for i in range(len(all_words)):
        totaltimestart = time.time()
        conversation_memory = ConversationBufferMemory()
        conversation_chain = None
        initialize_chain()
        conversation_memory.save_context({"input": prompt1()}, {"output": movieanalysis})
        
        
        word = all_words[i]
    
        #prompt_time, response = pick_word(word)
    
        ## first score
        memory = conversation_memory.load_memory_variables({}).get("history", "")
    
        for j in range(3):
            prompt_time, response = give_score(word, memory)
            words_list.append(word)
            score_list.append(response)
    
        total_time = time.time() - fulltime1
    
        #print("TOTAL TIME:", totaltimestart - totaltimeend)
    
    print("TOTAL GENERATION TIME:", total_time)

    
    
    return essay_time, total_time, movieanalysis, words_list, score_list

essay_time, total_time, movieanalysis, words_list, score_list = home(32)


# In[13]:


def extract_numbers(text):
    try:
        return float(re.findall(r'\d+\.?\d*', text)[0])
    except Exception:
        return -1

import pandas as pd

num_gpu = 32
data2 = {"gpu":[], "essaytime": [], "totaltime": [], "summary": [], "sentiment":[] }
df2 = pd.DataFrame(data2)
score_list_llama3 = []
for score in score_list:
    score_list_llama3.append(extract_numbers(score))
data = {"words": words_list, "scores": score_list_llama3}
df = pd.DataFrame(data)
df = df.sort_values(by = ["scores"], ascending = False)
ranked_words = df.drop(df[df.scores > 1].index).drop(df[df.scores < 0].index).groupby('words').agg('mean').sort_values(by = ["scores"], ascending = False)
top10 = str(ranked_words.index[0:10].tolist())[1:-1]
summarywords = len(movieanalysis.split(" "))

## returns words
relevant_words = str(ranked_words.index[0:10].tolist())[1:-1]
relevant_words




# In[ ]:




