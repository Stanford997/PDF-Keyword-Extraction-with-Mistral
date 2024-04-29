# PDF-Keyword-Extraction-with-Mistral

To extract the keyword from PDF, I use KeyLLM, an extension to KeyBERT for extracting keywords with Large Language Models like Mistral.

## 1. Data Preprocessing

First use PyPDF2 to extract text contents and the page number from pdf, then do the cleaning to remove characters like specific quotation marks.

## 2. Keyword Extraction with Mistral

### 2.1 Load the model

```python
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    model_type="mistral",
    gpu_layers=200,
    device_map="auto",
    hf=True,
    config=config
)
```

Here I load mistral 7B. It is a small language model but really accurate and instruct is a chat based model. `gpu_layers=200` offload some of its layers to be used on the GPU.

`config.config.context_length = 4096` Because the text contents in pdf are usually very long, so make sure to set this value very large.

```python
login(token="---")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

# Pipeline
generator = pipeline(
    model=model.to('cuda'), tokenizer=tokenizer,
    task='text-generation',
    max_new_tokens=50,
    repetition_penalty=1.1
)
```

For security, I hide the login token of my huggingface. Tokenizer is also from Mistral 7B and `max_new_tokens` is set to 50 to tell the model to only generate 50 new tokens at a max because we just want a number of keywords.

### 2.2 Prompt template

The prompt template is as follows (https://newsletter.maartengrootendorst.com/p/introducing-keyllm-keyword-extraction):

![img](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa648f3c4-d258-4455-b61a-afd8b17bdcda_1200x734.png)

- `Example prompt` - This will be used to show the LLM what a “good” output looks like
- `Keyword prompt` - This will be used to ask the LLM to extract the keywords

We can show the large language model an example of how it should extract keywords and then it repeats that. `Example prompt` shows how a conversation should go. After that, `Keyword prompt` essentially is the samething but this time we can plug in any document that we want and our documents pdf for which we wnat our keywords to be extracted.

### 2.3 KeyLLM

With the document and prompt, load them and model into KeyLLM

```python
# Load it in KeyLLM
llm = TextGeneration(generator, prompt=prompt)
kw_model = KeyLLM(llm)

for grouped_text in grouped_texts_list:
    keywords = kw_model.extract_keywords(grouped_text); 
    keywords_list = keywords_list + keywords
```

When the amount of text is particularly large, inputting the entire text directly into the model will cause the processing to be very slow, so here the text is input into the model in sections and a keywords list is obtained.

One problem with LLM extract keywords is that the keywords it outputs may not exist in the original text, so the last step is to remove these words and output the page number where each keyword is located.

