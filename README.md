# CS4120-final-project

### **Purpose Statement:** 

Our project introduces a text simplification tool that identifies challenging, uncommon words in a given text and replaces them with more commonly understood synonyms. This process preserves the original meaning and grammatical structure of the text, ensuring clarity and accessibility. By leveraging natural language processing techniques, our tool bridges gaps in language proficiency, enabling better comprehension for diverse audiences.


****
### **Data:** 

We used two datasets for our custom model: 

1. **Word Complexity Lexicon (WCL):** Dataset consisting of words and their complexity scores calculated by aggregating over human ratings. The score belongs to a scale of 1-6, where 1 represents "very simple" and 6 represents "very complex"
2. **Complex Word Identification Dataset (CWID):** Dataset with relevant information to us such as words, the number of native and non-native English speakers who found the word difficult to understand, and a probility score of someone finding that word difficult.

For our T5 model we used the following dataset: 

1. **WikiLarge Text Simplification:** Dataset containing pairs of "normal" and "simple" sentences.


****

### **Models:**

1. **Custom model:** We created our own simplification model from scratch using the following process

    _a) To predict out of vocabulary words, we tested neural nets and random forest and found that WCL performed best with neural nets, and CWID performed best with random forest._

    _b) We then tested 2 different Word2Vec models to get a list of similar words in a vector space to a word for replacement._

    _c) Put together, given a sentence, we identify words above a self-determined threshold and use the Word2Vec models to replace with similar words below that threshold._

2. **T5 Model:** We trained a T5 Model for text simplificaion to compare to our model

    _a) We finetuned the hyperparameters of our T5 model when trained on our text simplification dataset, and proceeded with the batch size, epochs, and learning rates that produced the least training and validation loss.

    _b) Given a sentence, the T5 model will output a simplified version. 
****

### **Evaluation:**

For our evaluation we used 2 metrics

1. **Flesch-Kincaid Reading Ease:** Metric that tests readability on a scale between 0 to 100

2. **ROUGE-L:** Metric that measures the overlap of n-grams with reference simplifications to test semantic preservation

****

### **Running the Application:**

For our project, we implemented an application where you can input a sentence and recieve a simplified version (with text-to-speech and speech-to-text included). To run this application, navigate to the text-simplification folder and follow the intructions in the README.md file

**** 
### **Viewing Model implementations:**

In the text-simplification folder, you can find the src directory which stores the script.py file where you can see the implementation of the custom model organized into a main function with classes for each dataset, as well as their simplification method. You can also find the T5 model implementation, as well as the evaluation metrics used for all of the models. 

In the home directory navigate to the src folder, and you will also find .ipynb files, where you can run all of these models individually. 

