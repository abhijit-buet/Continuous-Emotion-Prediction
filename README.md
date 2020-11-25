# Continuous-Emotion-Prediction
Continuous Emotion Prediction was a challenge launched in AVEC 2017. 
Three emotion dimension -arousal,valence and liking were required to predict.
We have trained a deep LSTM-RNN structure using Bag-of-Words features.
Here Different LSTM-RNN variants were used to train, including,
1.Bidirectional LSTM,
2.Multi tasking ,
3.Many to Many Mapping.
For evaluation, Pearson Correlation Coefficient(PCC) was used.
We got PCC about 0.562 for arousal, 0.543 for valence and 0.3512 for liking.
Our work got 10% improvement from the baseline.
