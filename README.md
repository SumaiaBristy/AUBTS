# AUBTS
Abstractive and unsupervised Bengali Text Summarization

Abstract: Our motivation for this work is based on the observation that while there has been significant research on abstractive text summarization, there are only a few models designed specifically for Bengali. However, the existing models either generate summaries with limited words or no new words, meaning containing only the words from the source texts. This can make it challenging to fully comprehend the original text. Additionally, the performance of abstractive summarization systems is dependent on a large collection of document-summary pairs, which is not available for low-resource languages like Bengali. Furthermore, the sequence-to-sequence models used in these systems may struggle with long input sentences, resulting in the loss of important information. Therefore, we propose a graph-based unsupervised abstractive summarization system for Bengali text documents that can generate summaries using a Part-Of-Speech (POS) tagger, a pre-trained language model trained on Bengali texts and one fine-tuned model. We use two readily available open source datasets from this work (ref-tasfeer) to evaluate our model. Additionally, we fine-tuned one pre-trained language model with the preprocessed datasets stated above. These two models can serve the users with the purposes for text summarization although our main focus is the unsupervised one and future plan is to work more on the supervised one.


**Referrenced Paper vs My Paper**

Title: Unsupervised Abstractive Summarization of Bengali Text Documents<br>
Authors: Dr. Dagmar Gromann, Prof. Sebastian Rudolph and Xiaoyu Yin<br>
My Name: Sumaia Aktar<br>
My Paper:  https://github.com/SumaiaBristy/AUBTS/blob/main/AUBTS_final_report.pdf [final_report.pdf]<br>
Institution: Brock University<br>
Department: Computer Science<br>
Course: 5P84 (Natural language Processing)<br>
My Supervisor: Ali Emami<br>

My work includes development of two bengali text summarization model in supervised and unsupervised setting. We need following experimental set up before runing the projects on windows operating system.
CPU @ 2.30GHZ
- 8GB RAM
- conda 23.3.1
- PyCharm – 2022.3.2
- Python 3.9.13
- torch 1.7+
- NetwokX 2.6
- FastAI (support torch below 1.8)
For the fine-tuned model you just need to run this file https://github.com/SumaiaBristy/AUBTS/blob/main/fineTuningParaphrasedSummary.py [** fineTuningParaphrasedSummary.py**] and for for running the corresponding file regarding unsupervised model, please read the experimental section of our paper https://github.com/SumaiaBristy/AUBTS/blob/main/AUBTS_final_report.pdf to get detailed idea for system configuration, problem set up, dataset preprocessing, traning argumnets and so on.
