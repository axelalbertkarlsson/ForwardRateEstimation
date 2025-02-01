Forward Rate Estimation using OIS Data

Overview
This project aims to estimate forward interest rate curves using Overnight Index Swap (OIS) data. The approach integrates Kalman filtering, Principal Component Analysis (PCA), and optimization techniques to construct a robust forward curve model. The model is designed to incorporate central bank policy meeting decisions and reduce dimensionality efficiently.

Methodology
Valuation of OIS Instruments

OIS contracts serve as proxies for the risk-free rate.
The floating leg is approximated using a geometric average of overnight interest rates.
Principal Component Analysis (PCA)

Used to extract the most significant risk factors influencing forward rate movements.
Dimension reduction is achieved by approximating the covariance matrix of the forward curves.
Optimization Model

The model optimizes state variables representing monetary policy decisions and PCA-extracted factors.
A Riccati recursion technique is applied to solve the state-space equations.
Expectation-Maximization (EM) is used to estimate noise parameters.
Interpolation Techniques

Various interpolation methods were tested, including:
Linear on discount factors
Linear on spot rates
Raw interpolation
Logarithmic rate interpolation
The impact of interpolation on forward curve accuracy was evaluated.
Key Results
Optimized Forward Curves: The model successfully estimates forward rates over a 10-year horizon while incorporating policy meeting adjustments.
Validation & Performance:
The Leave-One-Out Cross-Validation (LOOCV) method was used to assess the accuracy of forward curve predictions.
The model significantly reduces pricing errors while capturing economic trends effectively.
Comparison of Interpolation Methods:
Linear on Discount Factors showed the most stable forward curves.
Raw Interpolation provided more consistent estimates but lacked smoothness.
