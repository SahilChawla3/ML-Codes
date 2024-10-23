#Adaboost, Gradientboost, bagging, Random forest wale codes ke notes

'''
Test Size 

ye data ko testing and training me divide krta hai
test_size = 0.2
matlab 20% data test ke liye jaayega and 80% data training ke liye 


Test size machine learning mein data ka woh hissa hota hai jo model ko test karne ke liye alag rakha jaata hai. Yeh typically ek percentage hota hai, jaise 20% test ke liye aur 80% training ke liye. Iska maqsad yeh hota hai ki model ko unseen data par evaluate karke dekha ja sake ki wo nayi data par kitna accha predict kar raha hai
'''

'''
n_estimators

**n_estimators** machine learning mein un algorithms ke liye use hota hai jo **multiple models** ko mila kar kaam karte hain, jaise **Bagging**, **Random Forest**, aur **Boosting**.

### Aasaan Bhaasha Mein:
**n_estimators** ka matlab hota hai ki hum kitne **decision trees** ya **models** banayenge. Maan lo, agar tum **n_estimators = 100** set karte ho, toh tumhaara algorithm **100 decision trees** banayega. Fir, in sab trees ka combined result nikaal kar final prediction di jaati hai.

### Importance:
Jitne zyada estimators (trees) honge, model zyada powerful banega, lekin computation time bhi badh sakta hai. Zyada estimators se accuracy improve hoti hai, lekin ek limit ke baad fayda kam ho jaata hai.
'''

'''
random_state

**random_state** ek parameter hota hai jo machine learning mein **randomness ko control** karne ke liye use hota hai, taki tumhaare results **reproducible** ho sakein.

### Aasaan Bhaasha Mein:
Jab tum apna data randomly split karte ho (jaise training aur test sets me baantte ho), ya algorithms me randomness use karte ho (jaise decision tree banate ho), to har baar alag result aa sakta hai kyunki split ya randomness ka process har baar thoda alag hota hai. **random_state** ek fixed number (jaise 42) set kar deta hai, taaki har baar tumhaara code chalne par **same random split ya process** ho, aur tumhe same result mile.

### Example:
Agar tum **train_test_split** function me **random_state=42** set karte ho, to data hamesha ek hi tarike se split hoga, aur har baar exact same results milenge.
'''



#Decision Tree code notes

'''
### Decision Tree Criteria ka Difference:

1. **Gini Index**: Impurity ko measure karta hai; jitna kam Gini, utna behtar split. Yeh fast hota hai. Value 0 (pure) se 0.5 (impure) tak hoti hai.

2. **Entropy**: Uncertainty ko measure karta hai; jitna kam entropy, utna behtar split. Yeh information gain par based hota hai. Value 0 (pure) se 1 (impure) tak hoti hai.

3. **Log Loss**: Decision trees me splits ke liye nahi hota. Yeh model ki performance ko evaluate karta hai aur galat predictions ko zyada penalty deta hai. Logistic regression me zyada istemal hota hai.

Samjho: **Gini** aur **Entropy** decision tree me splits ke liye hain, aur **Log Loss** model ki accuracy check karne ke liye.
'''

#SVM Code Notes

'''
**Average** parameter ka use isliye hota hai kyunki jab aap multi-class classification karte hain (jismein 2 se zyada classes hoti hain), toh aapko precision ko calculate karne ke liye ek method chuni hoti hai. 

### Short Explanation of Average Types:

1. **macro**: Har class ki precision ko barabar importance deta hai, chahe unka size kaisa bhi ho. 
2. **micro**: Sab classes ke predictions ko combine karke ek overall precision nikaalta hai. 
3. **weighted**: Har class ki precision ko uske support (size) ke hisaab se weight karta hai.

In short, average parameter se aap specify karte hain ki precision kaise calculate kiya jaaye jab multiple classes ho.
'''

'''
SVC (Support Vector Classification) ek machine learning algorithm hai jo classification tasks ke liye use hota hai. Yeh Support Vector Machine (SVM) ka ek part hai.
'''

'''
### Linear Kernel vs RBF Kernel

1. **Linear Kernel**: Yeh simple hai aur sirf straight line (hyperplane) ka use karta hai data points ko separate karne ke liye. Yeh tab achha hai jab data linearly separable ho.

2. **RBF (Radial Basis Function) Kernel**: Yeh complex hai aur circular boundary create karta hai. Yeh data ko non-linear spaces me bhi separate kar sakta hai, isliye jab data linearly separable nahi hota tab yeh behtar hota hai.

In short, linear kernel straight line ke liye hai, jabki RBF kernel non-linear data ke liye use hota hai.
'''

'''
### Accuracy, Precision, Recall, F1 Score, and Confusion Matrix

1. **Accuracy**: Total correct predictions ka ratio hai (sahi predictions / total predictions). Yeh overall performance ko measure karta hai.

2. **Precision**: True positive predictions ka ratio hai (sahi positive predictions / total predicted positives). Yeh batata hai ki model ne jo positive predictions kiye, unme se kitne sahi the.

3. **Recall**: True positive ka ratio hai (sahi positive predictions / total actual positives). Yeh measure karta hai ki model ne kitne actual positive cases ko sahi se identify kiya.

4. **F1 Score**: Precision aur recall ka harmonic mean hai. Yeh un cases ke liye useful hai jahan precision aur recall ka balance zaroori hai.

5. **Confusion Matrix**: Ye ek table hai jo model ki performance ko summarize karta hai. Isme true positives, true negatives, false positives, aur false negatives ka count hota hai.

In short, accuracy overall performance ko dikhata hai, precision aur recall specific metrics hain, F1 score unka balance hai, aur confusion matrix detailed performance ko summarize karta hai.
'''

'''
### True Positive, True Negative, False Positive, and False Negative

1. **True Positive (TP)**: Jab model ne positive prediction kiya aur wo sahi hai (actual positive case). Example: Actual cancer patient ko cancer ka prediction dena.

2. **True Negative (TN)**: Jab model ne negative prediction kiya aur wo sahi hai (actual negative case). Example: Actual healthy person ko healthy ka prediction dena.

3. **False Positive (FP)**: Jab model ne positive prediction kiya lekin wo galat hai (actual negative case). Example: Healthy person ko cancer ka prediction dena.

4. **False Negative (FN)**: Jab model ne negative prediction kiya lekin wo galat hai (actual positive case). Example: Cancer patient ko healthy ka prediction dena.

In short, TP aur TN sahi predictions hain, jabki FP aur FN galat predictions hain.
'''

#RMSE Code Notes

'''
### RMSE (Root Mean Squared Error)

**RMSE** ek metric hai jo model ki prediction errors ko measure karta hai. Yeh actual values aur predicted values ke beech ka average error ko dikhata hai. 

- **Formula**: RMSE = √(Σ(predicted - actual)² / n)  
- **Purpose**: Iska use model ki accuracy ko evaluate karne ke liye hota hai; jitna kam RMSE, utna behtar model.

In short, RMSE model ki predictions ki galtiyon ka ek standard measure hai.
'''

#PCA Code Notes

'''
### PCA (Principal Component Analysis)

**PCA** ek dimensionality reduction technique hai jo data ki complexity ko kam karne ke liye use hoti hai. Iska main goal hai high-dimensional data ko lower dimensions me project karna, bina zyadatar information khoe.

- **Kaise Kaam Karta Hai**: PCA data ke variance ko analyze karta hai aur usko principal components (new features) me transform karta hai. Yeh components data ke sabse zyada variance ko capture karte hain.

- **Use Cases**: PCA ka use visualization, noise reduction, aur machine learning me features ki number ko kam karne ke liye hota hai.

In short, PCA ek technique hai jo high-dimensional data ko simplify karne aur uske structure ko samajhne me madad karti hai.
'''