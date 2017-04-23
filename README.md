# A Two-Fold Exploratory Study on AES


[![Build Status](https://travis-ci.org/fchollet/keras.svg?branch=master)](https://github.com/alivcor/aes)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/alivcor/aes/blob/master/LICENSE)

## What is this project about
Automated Essay Scoring systems are being widely used in the industry - The ETS uses AES to grade the AWA section of the new GRE, and same applies to the GMAT. The present research has been limited to rely heavily on extracting care- fully designed features to evaluate and score essays through training on huge datasets. This makes it impossible for primary school teachers to use such systems for grading. Moreover, until recently, even the systems which involve training on huge datasets yielded average results. In this project, we want to study both perspectives of solving this problem.

<p align="center">
<img src="https://github.com/alivcor/aes/blob/master/DeepScore/deepscore_logo_png.png" alt="DeepScore" height="300" style="display: block; margin: 0 auto;" align="middle"/>
</p>

<br/>

<p align="center">
<img src="https://camo.githubusercontent.com/abb87e84f23816282f054ac08886a451518dacdf/68747470733a2f2f626c6f672e6b657261732e696f2f696d672f6b657261732d74656e736f72666c6f772d6c6f676f2e6a7067" alt="DeepScore" height="100" style="display: block; margin: 0 auto;" align="middle"/>
</p>

## Goals
1. Perform extensive feature engineering to find out specific cues to grade essays in the cases where sample size could be as low as 5 graded essays.
2. Implement the state-of-the-art system using Recurrent Neural Networks for grading essays - this involves training on conventional datasets.

## Implementation level details
We will be using the ”The Hewlett Foundation: Automated Essay Scoring” dataset from Kaggle by The Hewlett Foun- dation which consists of around 1785 essays on 8 different topics, in a score range of 0-6.

1. The first part of our project will be to implement a basic Machine Learning Model described in (1)
2. The second part will involve improving the above model by creating a pipeline of modules which would give us several high-level features, and trying to make our model achieve the same level of QWK (accuracy in grading) as the model in the first part, using just 4-5 examples using methods described, but not limited to (4). This would conclude the first fold of our study.
3. For the last part, we will implement a neural network based system described in (2) - DeepScore


## Modules
1. Statistical Analysis Module: - Baseline features such as Word Count, Long Word Count, Sentence Count, Paragraph Count, Average Paragraph Length.
2. Semantic Analysis Module - This will primarily involve using Latent Semantic Analysis to find out the semantics of the essay.
3. Syntactic Analysis Module: - This module will have sev- eral functions. It will perform Part-of-Speech tagging for ex- tracting syntactic features, find out mistakes in syntax and grammar, incomplete sentences. It will also try to extract concepts that a particular essay is trying to convey.
4. Clause Analyzer - This will try to identify the main clause in the essay. Next would be to identify Subordinate Clauses, Relative Clauses and Essential Relative Clauses, the infini- tive and the compliment.
5. DiscourseAnalysisModule:-Thismodulewillcapturethe organization of ideas, and flow (with /against). It will identify thetoneandthesideorflowonascaleof-3to3forthewhole essay. A tricky part would be to find out ”flips”: Identify if the author has flipped tone within the essay at any point by partitioning essay into separate arguments.
6. Topical Analysis Module: - This module would try to cap- ture richness in vocabulary. As given in (5) it will try to cap- ture ”trins” by finding the most closely related ideas to any essay in the training set.

## Tools
* The implementation will be done in python. We will use NLTK, SciPy, NumPy and sklearn libraries. Other than that, we will be using Keras on top of TensorFlow.
* We will evaluate our model using Quadratic Weighted Kappa which measures inter-rater agreement for qualitative (categorical) items on test set.

## Relevant Work
1. Manvi Mahana, Mishel Johns, Ashwin Apte Automated Essay Grading Using Machine Learning - Stanford Uni- versity
2. Kaveh Taghipour and Hwee Tou Ng A Neural Approach to Automated Essay Scoring Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (Association for Computational Linguistics 2016)
3. Jill BURSTEIN and Daniel MARCU Benefits of Modu- larity in an Automated Essay Scoring System ISI, Uni- versity of Southern California (Association for Compu- tational Linguistics 2000)
4. George Forman and Ira Cohen Learning from Little: Comparison of Classifiers Given Little Training Hewlett- Packard Research Laboratories 1501 Page Mill Rd., Palo Alto, CA 94304
5. Mark D. Shermis and Jill C. Burstein Automated Essay Scoring - A Cross Disciplinary Perspective Routledge, 2002

## Future Work
1. Research on how feature engineering coupled with RNNs can allow even better results.
2. Performing a user-study involving volunteers to write es- says and then grading it using our system, comparing it with human graded scores.
