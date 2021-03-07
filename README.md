# project_3
# BC_Project3 : LEAPS and Bounds

## Background  

Long-term equity anticipation securities (LEAPS) are publicly traded options contracts with expiration dates that are longer than one year, and typically up to three years from issue. Trading LEAPS offers the leverage capability options contracts, but because of their long-dated expiries, 
The goal of this research project is to examine techniques and tools for different strategies centered around the trading of Long-term equity anticipation securities (LEAPS), utilizing various data points and technical indicators.


## Approach: strategies and tools  

*   Construct functions with changing variables
*   Use iteration to apply the functions to wide sets of data, apply different parameters to the options we dealt, and different approaches to the signals, in order to determine whether strategies could be formed across the range of indicators used.
*   Use and construct a number of different strategies
*   Construct a number of buying strategies by utilizing technical indicators such as Bollinger Bands,  Volume –both the underlying/options, and Put/Call ratios
*   Combine LSTM and Random Forest models –in essence, themselves, technical indicators

##  Libraries and Tools  
*   Google Collab  
    <https://colab.research.google.com/notebooks/intro.ipynb>
*   Iter Tools and Functions - Functions creating iterators for efficient looping  
    <https://docs.python.org/3/library/itertools.html>
*   Streamlit - the fastest way to build and share data apps  
    <https://streamlit.io/>
*   ML Libraries: RF and LSTM
*   Data Source: Yahoo Finance and Discount Option Data

## Code
* To navigate the code, start with get_data, which sets a directory path and goes through every file with a csv ending. 
* It will then append the row to a bin if it meets the symbol and date criteria. The bins are then added to the dictionary, and 
*the dictionary will be turned to a dataframe saved to a pickle.
* We then process the pickle for various features and ML models, to start creating signals and help with option selection
* With signals and option selection perimeters in place, we choose the calls or puts dataframe for a symbol, then append the profit
* or loss of the value where there is a signal.

## Statistics  

*   Option data available for 4500 different symbols organized by day!
*   We narrowed it to 25 symbols grouped by six sectors over four years
*   We ran five different strategies using nine different compositions of signals, with nine different parameters for options selection
*   Data is MASSIVE! 120GB, 900 million lines in the csv files  

## Process

*   Create .csv files for each sector with the data sets mentioned –price, technicals, etc.
*   Take the data for each sector/name, and apply the various strategies we built, and then log the returns
*   With the results, we intend to gain insight into risk factors regarding each strategy  

##  Example strategies

*   Selling Put Credit Spreads
*   LEAPS
*   LEAPS/Call Calendar Spreads
*   Underlying
*   Underlying/Covered (Short) Calls
*   Underlying/Selling (Short) Puts  

## Stremlit application and graphs

*   We used streamlit to diaplay a dashboard where the user can select one or more strategies to evalaute investing strategies based on a choice of the sector and symbol under that sector.
*   Execute the following command to run the app locally  

    ```shell
    streamlit run mstrategies.py
    ```

    * You should see the following dashboard. Select the `Sector` and `Symbol` follwoed by differnt strategies to get graphs plotted on the right hand side
    ![Streamlitapp](Images/Leapsandbounds.png) 

    *   A sample graph for JPM is shown below:  

        ![JPMctmleaps](Images/JPM_ctm_leaps.png)  

        ![JPMftmleaps](Images/JPM_ftm_leaps.png)

    * For a brief clip on the app, please play the following link  
    
        <https://drive.google.com/file/d/1KMXna3p0Ieii2r7A-BUTCLA5B3q_Z6hT/view?usp=sharing>

## Insights and challenges

*   It Pays to Pay for Big Data!
    *   Don’t Use Google Drive
    *   AWS Sagemaker seems to be a better alternative

*   Computational Limitations
    *   Corrupted Data
    *   Long Run Times/Delays with respect to Trial and Error


*   Lessons Learned
    *   Smaller Data Sets
    *   Perhaps Less Breadth and Scale  

##  Further improvement

*   With more time, the goal would be to do more to fine-tune to calibrate variables, as well as think through the scenarios that detect the best fit for each strategy

*   Due to the scale data used, along the way, we encountered a number of obstacles with regard to massaging the data

*   We faced a number of challenging issues involving the code –also do to the breadth















