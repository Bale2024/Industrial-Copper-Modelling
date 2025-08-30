# Industrial-Copper-Modelling
The project deals with the challenges of industry data (Copper) related to sales and pricing. The aim of the project is to predict status while capturing leads and selling price of the Copper based on the User input. The challenges are solved using Machine Learning techniques (Classification and Regression).

Libraries Required:
1. Dashboard Creation:
* Streamlit
2. Calculation :
* Numpy
* Pandas
* Scipy
3. Data Visualization:
* Seaborn
* Matplotlib
4. MAchine Learning:
* Scikit-learn
* imblearn
* Random Forest Classifier
* XGB Classifier
* Logistic Regression
* Decision Tree Regressor
* XGB Regressor
* Extratrees Regressor

Tools Required:
* Visual studio code
* Python  
```
Process:
* The link of the raw-Dataset (Excel file) will be provided in  the problem_statement of the project
* Load the Dataset, perform necessary transformation on the file and make into a suitable format.
* Performed Pre-processing steps
  
    1.Handling Missing Values and Imputing with Mean, Median and mode values.

    2.Checked for Duplicate Rows in the dataset.

    3.Identified outliers using Boxplot and treated them usiing IQR method.

    4.Skewness in the numerical features is identified and treated using Box-cox Transformation.

    5.Visualized the Outliers and skewness before and after treatment with KDE plots, Boxplots.

    6.Co-Relation is performed using a Heatmap for the numerical features in the dataset.

    7.Encoding is performed on the Categorical features as per the requirement.

    8.Converted the Delivery date and Item date into the date time format and Extracted year as part of feature Engineering.
 ```   
* The dataset is now split into train and test cases, as a part of ML pipeline.
* The Classification and Regression ML modes are trained with the train data.
* For Predicting the STATUS and SELLING_PRICE(Target Features), the models will be evaluated on the test data.
* EValuation metrics like Accuracy,Precision, Recall, F1 Score are calculated using Classification Report.
* The ROC-AUC Curves are generated for the different models used, and R2-score are calculated.
* For Classification task, Random forest Classifier was chosen since it had a high accuracy of 92.5%
* For Regression task, XGBRegressor outperformed the other models with R2-score of 0.88(closer to 1) and hence considered as a best fit.
* The best models and encodings are pickled using pickle.dump method. 
* As a part of Front End Visualization, An app has developed using Streamlit where User will provide inputs to predict status and Selling_price.



  
