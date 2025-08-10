# Python Topic Modelling for Sentiment Analysis

This project implements sentiment classification on the Sanders dataset using both Bag-of-Words (BOW) and Latent Dirichlet Allocation (LDA)-based feature extraction methods. The experiments evaluate multiple classifiers (Decision Tree, Naive Bayes, K-Nearest Neighbors) on different feature sets (E40, E100, E300, E500, and EBOW).

## Dataset
- **Name**: Sanders Twitter Sentiment Dataset  
- **Labels**: Positive, Negative, Neutral  
- **Preprocessing**:
  - Text normalization  
  - Tokenization  
  - Stopword removal  
  - Lemmatization  
- **Note**: Due to licensing, the dataset is not included in this repository. Please download it from the official source or use your own equivalent dataset.

## Requirements
Install dependencies via:
```bash
pip install -r requirements.txt


Follow these steps to run your Python notebooks (BOW.ipynb and FE+BOW.ipynb) in Google Colab:
Step 1: Upload the Notebook to Google Colab
Open Google Colab in your web browser.
On the welcome page, click on the File tab in the top left corner.
From the dropdown, select Upload notebook.
A file selection dialog will appear. Browse to the location where your .ipynb file is saved.
Select BOW.ipynb or FE+BOW.ipynb and click Open.
This will upload the notebook into Google Colab, where you can view and execute the code.
Step 2: Run the Code in the Notebook
Once the notebook is open, you can run the cells one by one by clicking the Play button to the left of each code cell.
Alternatively, you can run all cells in the notebook at once by selecting Runtime from the top menu and choosing Run all.
Step 3: Save Your Work
To save any changes you make to the notebook:
Click on File > Save a copy in Drive. This will save your work to Google Drive for future reference.
