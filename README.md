# Continuous-Emotion-Prediction
Continuous Emotion Prediction was a challenge launched by AVEC 2017. 
Three emotion dimensions -arousal,valence and liking were required to predict.

<h1> Proposed Method
<h6>
We trained a deep LSTM-RNN structure using Bag-of-Words features.
Here Different LSTM-RNN variants were used to train, including,
1.Bidirectional LSTM,
2.Multi tasking ,
3.Many to Many Mapping.
<img src="https://github.com/abhijit-buet/Images/blob/main/Slide3.PNG" width="768" height="512">



<h1> Result
 <h6>
For evaluation, Pearson Correlation Coefficient(PCC) was used.
We got PCC about 0.562 for arousal, 0.543 for valence and 0.3512 for liking.
Our work got 10% improvement from the baseline.
