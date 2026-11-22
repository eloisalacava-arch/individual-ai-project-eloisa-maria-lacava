# Project Plan — IMDb Sentiment Analysis

## 1. Project Idea
**Can AI Really Understand Movie Reviews? A Study of Sentiment Classification Errors**

We will investigate how accurately an AI model can classify IMDb movie reviews as positive or negative, and identify the situations in which it makes mistakes.

**Dataset:** Hugging Face — `stanfordnlp/imdb` (50,000 labelled reviews).

## 2. Research Question
How accurately can AI classify movie reviews as positive or negative, and which characteristics of the reviews are associated with classification errors?

## 3. Theory
AI models should correctly classify most reviews when the language expresses a clear positive or negative opinion.

However, reviews containing sarcasm, mixed opinions, ambiguity, or complex language may be harder to classify.

**Our hypothesis:** AI classification accuracy will be lower for ambiguous or linguistically complex reviews than for straightforward reviews.

## 4. What Would Prove Our Theory Wrong?
Our hypothesis would be challenged if reviews containing sarcasm, mixed opinions, ambiguity, or complex language have similar or higher classification accuracy than straightforward reviews.

To test this, we will define these categories before evaluating the model, compare its predictions with the dataset labels, and independently verify the results.
