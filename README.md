# Automated Intelligent Analyzer of Electrochemical Impedance Spectroscopy (AIA-EIS)
## Vision
Build a web-based, scientific, open, automated EIS analyze platform. Related paper: https://www.sciencedirect.com/science/article/abs/pii/S0013468622005126. A video report in chinese is recored in https://www.bilibili.com/video/BV1db4y1v7wY.

## Supervisors
### Main Supervisor-1
Prof. Ying Jin
### Main Supervisor-1
Prof. Peng Shi

## Contributors
### Main Contributor-1
Zhaoyang Zhao Email: zzhaoyang2023@gmail.com, Researchgate: https://www.researchgate.net/profile/Zhaoyang-Zhao-2; 
Zhaoyang is about to get his Ph.D. and looking for a postdoctral position related with elctrochemical, machine learning, atomic force spectroscopy. If you are interested in this work, cooperation is always possible.
### Main Contributor-2
Lianheng Zou
### Main Contributors
Hui Zhi neu3dmaker@163.com; Peng Liu. Yang Zou

## Main Function
###  1- Automated EIS outlier detection.
For the first time, an EIS outlier definition consisting of three qualitative conditions was proposed, which was then taken as a foundation and combined with the improved linear Kramers-Kronig validation, smoothing algorithm, quartiles, and a few sensitive residuals to build an automated workflow to automatically detect and remove outliers. 
Related Code is: aia_eis/v0/aia_eis_v0/a_d
Related Published Work: https://www.sciencedirect.com/science/article/pii/S0013468623008393

###  2- ECM prediction using machine learning model
An Interpretable Adaboost model (a kind of machine learning algoriithm) predicts an equivalent circuit model (ECM) for a given EIS
Related Code is: aia_eis/v0/aia_eis_v0/ml_sl
Related Published Work: https://www.sciencedirect.com/science/article/abs/pii/S0013468622005126

###  3- Automated fitting of  the parameters of the ECM.
Global optimization algorithm is applied to fit the parameters of the ECM.
Related Code is: aia_eis/v0/aia_eis_v0/goa
Related Published Work: https://www.sciencedirect.com/science/article/abs/pii/S0013468622005126

If you adopt the dataset or code in a scientific work, please cite related papers.

## Versions
### V0
#### Status
Still in construction.
The web platform is based on Flask MVC Framework and will be online around 2024.01.

## License
This project is licensed under the terms of the MIT license.
