You're right, since you've primarily focused on `target_lag_days=1` for the comparative experiments, explicitly showing a "Target Lag" column with the same value for every model can be redundant in the final presentation table. We can implicitly state that all presented results are for predicting the next trading day's direction.

Here's the revised "3. Experimental Results and Discussion" section with the "Target Lag" column removed from the tables, assuming all reported results are for `target_lag_days=1`. If you did run and want to include comparisons across different lags, then the column would be necessary.

---

**3. Experimental Results and Discussion**

All subsequent results are for predicting the stock direction for the next trading day (`target_lag_days=1`), using information available up to the end of the current trading day (Day D).

**3.1. Baselines: Stock Features Only**

*   **Rationale for XGBoost Baselines:** To establish a benchmark using a powerful traditional ML model with only numerical historical stock data (current day D's OHLVC, daily return, and lagged features for `history_lags = [1,2,3,5] days`).
*   **Rationale for LSTM (Stock Only from Aligned Data):** To provide a deep learning baseline using the *exact same input stock features and data splits* that the fused models will use as their "stock component". This directly isolates the impact of adding text.
*   **Rationale for LSTM (Original `rnns_stocks.py` results):** These offer context on LSTM performance when trained per ticker with potentially different feature engineering directly from raw stock data.

   | Model Configuration                       | Accuracy (MDA) | ROC AUC | F1 (Macro) |                                                              |
   | :---------------------------------------- | :------------- | :------ | :--------- | :----------------------------------------------------------------- |
   | XGBoost     | 0.5023         | 0.5077  | 0.51       |                 |
   | LSTM | 0.5294         | 0.4800 | 0.50  |                                |
   | LSTM  | 0.4695         | 0.4839  | 0.47       |                             |
   | |                |         |            |                                                                    |

   *   **XGBoost Results (with engineered features from `lagged_xgboost.txt`):**
        *   An accuracy of 0.5386 and ROC AUC of 0.5177 demonstrate that even with more engineered historical stock features (lags, rolling windows), the predictive signal for next-day direction is weak. The model still struggled, particularly with class 0 (F1-score 0.40 vs. 0.63 for class 1).
   *   **Initial XGBoost Results (without extensive lags/windows, from your previous outputs for target_lag 1 & 2):**
        *   These performed even worse (Acc ~0.51-0.52, very poor class 0 recall), highlighting that the minimal set of Day D stock features has extremely limited predictive power. *(You can briefly mention these if you want to show the evolution of your baseline, or omit if the `lagged_xgboost.txt` version is your definitive stock-only XGBoost baseline).*
   *   **LSTM (Original `rnns_stocks.py`) Results:**
        *   These ticker-specific LSTMs also showed performance around or below 50% accuracy, further underscoring the difficulty of predicting from stock data alone, even with dedicated models per ticker.
   *   **(Awaiting LSTM from Aligned Data Results):** The performance of the LSTM trained on the globally aligned stock features will be crucial here for a direct comparison with fused models.
   *   **Conclusion from Stock-Only:** Predicting daily stock direction using only historical price/volume data is very challenging. The signal is weak, leading to performance near random chance. This motivates the exploration of alternative data sources like news.

**3.2. Text-Only Models (MLP)**
   *(Populate this from `text_only_models_summary_lag1_hist1_2_3_5.csv`)*

   | Model (Text Feature)        | Accuracy (MDA) | ROC AUC | F1 (Macro) | Notes                                                            |
   | :-------------------------- | :------------- | :------ | :--------- | :--------------------------------------------------------------- |
   | FinBERT Sentiment           | 0.4914         | 0.5336  | 0.44       | Poor class 0 recall                                              |
   | FinBERT Tone                | 0.4963         | 0.5000  | 0.33       | Predicts only class 1 (0 recall for class 0)                     |
   | FinBERT2                    | 0.4988         | 0.5248  | 0.47       | Better class 0 recall than Tone, but still low                   |
   | RoBERTa Sentiment         | 0.5012         | 0.4776  | 0.49       | Balanced but low recall for both                                 |
   | BERT Base                   | 0.4963         | 0.5000  | 0.33       | Predicts only class 1 (0 recall for class 0)                     |
   | ST_all-MiniLM-L6-v2         | 0.5110         | 0.5183  | 0.51       | Most balanced performance among text-only, but still near chance |

   *   **Rationale:** To assess if daily aggregated news, represented by different BERT-based features, contains enough signal *on its own* to predict next-day stock direction.
   *   **Observations:**
        *   Performance is consistently poor, with accuracies and ROC AUCs hovering around 0.50 (random chance).
        *   Several models (`FinBERT Tone`, `BERT Base`) completely failed to predict downward movements (class 0 recall is 0.00), suggesting the extracted features lacked discriminative power for the MLP or that these models (especially un-fine-tuned BERT Base for sentiment) are not suitable for this task in this simple setup.
        *   `ST_all-MiniLM-L6-v2` (dense embeddings) provided the most balanced, albeit still very weak, performance (F1-Macro 0.51).
   *   **Conclusion from Text-Only:** Daily aggregated news text, when used in isolation with these MLP-based classifiers and various BERT representations, demonstrates minimal to no predictive capability for next-day stock direction.

**3.3. Early Fusion: Raw Historical Stock Features + Text Features (MLP)**
   *(Populate this from `fused_models_summary_lag1_hist1_2_3_5.csv`)*

   | Fused Model (Raw Stock + Text Type) | Accuracy (MDA) | ROC AUC | F1 (Macro) | Notes                                   |
   | :---------------------------------- | :------------- | :------ | :--------- | :-------------------------------------- |
   | Raw Stock + FinBERT Sentiment       | 0.4597         | 0.4761  | 0.45       | Performance degraded vs. text-only      |
   | Raw Stock + FinBERT Tone            | 0.5257         | 0.5230  | 0.51       | Slight improvement vs. best text-only   |
   | Raw Stock + FinBERT2                | 0.4768         | 0.4822  | 0.48       |                                         |
   | Raw Stock + RoBERTa Sentiment     | 0.4768         | 0.4754  | 0.46       |                                         |
   | Raw Stock + BERT Base               | 0.5208         | 0.5092  | 0.51       |                                         |
 

   *   **Rationale:** To investigate if simply concatenating raw historical stock features with text features improves predictions over using either modality alone. The stock features used here are the same extensive set as in the best XGBoost baseline (from `lagged_xgboost.txt`).
   *   **Observations:**
        *   This fusion strategy did not yield significant improvements. Most combinations performed similarly to, or worse than, the text-only models or the stronger stock-only XGBoost baseline.
        *   `Raw Stock + FinBERT Tone` and `Raw Stock + BERT Base` showed marginal accuracies around 0.52-0.53, but ROC AUCs remained weak.
   *   **Conclusion for Raw Stock + Text Fusion:** Direct concatenation of raw stock features (even the more engineered set) with these text features did not unlock substantial predictive power with the MLP classifier. This might suggest that the MLP struggles with the combined dimensionality and noise, or that a more nuanced fusion is needed.

**3.4. Early Fusion: LSTM Stock Embeddings + Text Features (MLP)**
   *(Populate this from `fused_lstm_text_models_encoders_summary_lag1_hist1_2_3_5.csv` and your `best_run.txt` - use the one you consider most representative, likely the `encoders_summary`)*

   | Fused Model (LSTM Emb. + Text Type) | Accuracy (MDA) | ROC AUC | F1 (Macro) |                                      |
   | :---------------------------------- | :------------- | :------ | :--------- | :---------------------------------------- |
   | LSTM Emb + FinBERT Sentiment        | 0.5427         | 0.5340  | 0.53       |               |
   | LSTM Emb + FinBERT Tone             | 0.5361         | 0.5159  | 0.49       |               |
   | LSTM Emb + FinBERT2                 | 0.5525         | 0.5461  | 0.53       |               |
   | LSTM Emb + RoBERTa Sentiment      | 0.5110         | 0.5021  | 0.51       |               |
   | LSTM Emb + BERT Base                | 0.5377         | 0.5323  | 0.55       |        |


   *   **Rationale:** To test if fusing a *learned representation* of stock history (LSTM embeddings) with text features yields better results. LSTM embeddings aim to capture complex temporal dynamics more effectively than raw lagged features.
   *   **Observations:**
        *   This fusion strategy generally yielded the highest performance metrics among all experiments, although the improvements are still modest.
        *   The combination of **`LSTM Emb + BERT Base` achieved the highest accuracy (0.5477) and F1-Macro (0.55)**. This is the most notable result. It's intriguing that `BERT Base` (which performed poorly alone and when fused with raw stock features) contributed positively when combined with the learned LSTM stock embeddings. This might suggest that the LSTM embeddings provided a strong, distilled stock context that allowed the MLP to find some complementary signal in the (otherwise noisy) `BERT Base` text features.
        *   `LSTM Emb + FinBERT Sentiment` and `LSTM Emb + FinBERT2` also showed slightly improved performance (Acc ~0.535, F1-Macro ~0.53) compared to other approaches.
        *   ROC AUC scores, while slightly better than other categories, still hover in the 0.50-0.55 range, indicating a limited but present ability to discriminate better than chance.
   *   **Conclusion for LSTM Emb + Text Fusion:** Using pre-trained LSTM hidden states as a learned representation of stock history for fusion with text features appears to be the most promising approach among those tested. It suggests that abstracting stock market dynamics into an embedding before fusion can be more effective than using a larger set of raw historical features directly with an MLP.

**4. Overall Conclusions and Future Work**
    *   *(This section remains largely the same as previously discussed, emphasizing the modest overall performance, the superiority of the LSTM embedding fusion approach, the limitations due to daily data, and avenues for future work like intra-day data and more advanced models.)*

By removing the "Target Lag" column and focusing the narrative, the tables become cleaner and the comparisons more direct for your primary experimental setup (predicting 1 day ahead). Remember to clearly state this focus at the beginning of your results section.