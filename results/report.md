**2. Methodology**

Our experimental pipeline begins with meticulous data alignment. Daily stock data (Open, High, Low, Close, Volume) sourced from Yahoo Finance and daily financial news headlines and summaries from Alpha Vantage were aligned by ticker and date. For each trading day (Day D) and ticker, all news items published on that day were aggregated into a single text document. The prediction target was defined as the binary direction of the stock's closing price on the next trading day (Day D+1) relative to its closing price on Day D.

Feature engineering involved several streams. For stock-based models, historical features included current Day D's OHLVC, its daily return, and lagged values (1, 2, 3, and 5 days prior) of closing price, daily return, and volume (as used in `lagged_xgboost.txt` and the LSTM stock-only baseline). For text-based models, various BERT-variant encoders were employed to transform the aggregated daily news text. These included domain-specific models like `ProsusAI/finbert` ("FinBERT Sentiment"), `yiyanghkust/finbert-tone` ("FinBERT Tone"), and `Narsil/finbert2` ("FinBERT2") for sentiment scores; a general robust sentiment model `cardiffnlp/twitter-roberta-base-sentiment-latest` ("RoBERTa Sentiment"); a general `bert-base-uncased` ("BERT Base"); and `all-MiniLM-L6-v2` via SentenceTransformers for generating dense semantic embeddings. In a more advanced fusion approach, pre-trained LSTMs (originally from `rnns_stocks.py`, trained on historical stock data with features including lags and windows) were used to generate "LSTM stock embeddings" by extracting their final hidden states for Day D.

The predictive models explored included XGBoost for a traditional machine learning baseline, a simple LSTM architecture for a deep learning stock-only baseline using the aligned stock features, and Multi-Layer Perceptrons (MLPs) for text-only and all fused model configurations. Early fusion was primarily achieved by concatenating scaled stock features (either raw historical or LSTM embeddings) with scaled text features. All models were trained and evaluated using a chronological 80/20 train/test split (or 70/15/15 train/val/test for MLPs). Numerical features were standardized using `StandardScaler`. Performance was assessed using Accuracy (Mean Directional Accuracy), ROC AUC, F1-Score (Macro and per class), RMSE, and Confusion Matrix.

**3. Experimental Results and Discussion**

All subsequent results are for predicting the stock direction for the next trading day (`target_lag_days=1`).

**3.1. Baselines: Stock Features Only**

*   **Rationale:** To establish benchmark performance using only historical stock market data. The XGBoost model used an extensive set of engineered features (current day + lags + rolling windows), while the "LSTM (Aligned Data)" used current day + simple lags. The "LSTM (Original rnns_stocks.py)" results provide context from a ticker-specific LSTM with different feature engineering.

   | Model Configuration                       | Accuracy (MDA) | ROC AUC | F1 (Macro) | Notes                                                        |
   | :---------------------------------------- | :------------- | :------ | :--------- | :----------------------------------------------------------- |
   | XGBoost (Eng. Hist. Stock Feat.)          | 0.5386         | 0.5177  | 0.51       | From `lagged_xgboost.txt`                                    |
   | LSTM (Aligned Hist. Stock Feat.)        | 0.5023         | 0.5077  | 0.50       | LSTM Baseline from Aligned Data                              |
   | LSTM (Original `rnns_stocks.py` - AAPL) | 0.4695         | 0.4839  | 0.47       | Context only; different training setup (others similar)      |

   *   **Observations:** The XGBoost model, with more comprehensive feature engineering including rolling windows and lags, achieved an accuracy of 0.5386. This is marginally better than random chance but indicates the inherent difficulty in predicting next-day movements even with detailed historical stock data. The LSTM trained on the globally aligned set of simpler historical stock features performed at 0.5023 accuracy, essentially at chance level. The original ticker-specific LSTMs from `rnns_stocks.py` (e.g., AAPL at 0.4695 accuracy) also struggled.
   *   **Conclusion from Stock-Only:** Predicting daily stock direction using only historical price/volume data is very challenging. The signal is weak, leading to performance that is, at best, slightly above random chance. This underscores the need to explore alternative data sources like news.

**3.2. Text-Only Models (MLP)**

*   **Rationale:** To assess if daily aggregated news, represented by different BERT-based features, contains enough signal *on its own* to predict next-day stock direction.

   | Model (Text Feature)        | Accuracy (MDA) | ROC AUC | F1 (Macro) | Notes                                                            |
   | :-------------------------- | :------------- | :------ | :--------- | :--------------------------------------------------------------- |
   | FinBERT Sentiment           | 0.4914         | 0.5336  | 0.44       | Poor class 0 recall                                              |
   | FinBERT Tone                | 0.4963         | 0.5000  | 0.33       | Predicts only class 1                                            |
   | FinBERT2                    | 0.4988         | 0.5248  | 0.47       | Low class 0 recall                                               |
   | RoBERTa Sentiment           | 0.5012         | 0.4776  | 0.49       | Low recall for both classes                                      |
   | BERT Base                   | 0.4963         | 0.5000  | 0.33       | Predicts only class 1                                            |
   | ST_all-MiniLM-L6-v2         | 0.5110         | 0.5183  | 0.51       | Most balanced among text-only, but near chance                   |

   *   **Observations:** Performance is consistently poor, with accuracies and ROC AUCs hovering around 0.50. `FinBERT Tone` and `BERT Base` (without specific sentiment fine-tuning for this output head) completely failed to predict downward movements. `ST_all-MiniLM-L6-v2` (dense embeddings) provided the most balanced, albeit still very weak, performance.
   *   **Conclusion from Text-Only:** Daily aggregated news text, when used in isolation with these MLP-based classifiers and various BERT representations, demonstrates minimal to no predictive capability for next-day stock direction. The aggregation or daily resolution likely dilutes any specific signals.

**3.3. Early Fusion: Raw Historical Stock Features + Text Features (MLP)**

*   **Rationale:** To investigate if simply concatenating raw historical stock features (the extensive set from the best XGBoost baseline) with text features improves predictions over using either modality alone.

   | Fused Model (Raw Stock + Text Type) | Accuracy (MDA) | ROC AUC | F1 (Macro) | Notes                                   |
   | :---------------------------------- | :------------- | :------ | :--------- | :-------------------------------------- |
   | Raw Stock + FinBERT Sentiment       | 0.4597         | 0.4761  | 0.45       | Degraded vs. text-only                  |
   | Raw Stock + FinBERT Tone            | 0.5257         | 0.5230  | 0.51       | Marginal improvement                    |
   | Raw Stock + FinBERT2                | 0.4768         | 0.4822  | 0.48       | No clear benefit                        |
   | Raw Stock + RoBERTa Sentiment       | 0.4768         | 0.4754  | 0.46       | No clear benefit                        |
   | Raw Stock + BERT Base               | 0.5208         | 0.5092  | 0.51       | Marginal improvement                    |
   | Raw Stock + ST_all-MiniLM-L6-v2     | 0.4914         | 0.5133  | 0.49       | Degraded vs. text-only MiniLM           |

   *   **Observations:** This fusion strategy generally did not yield significant improvements. Most combinations performed similarly to, or worse than, text-only models. The best results here (`Raw Stock + FinBERT Tone` at 0.5257 accuracy) were only marginally better than chance and did not surpass the stock-only XGBoost baseline.
   *   **Conclusion for Raw Stock + Text Fusion:** Direct concatenation of raw historical stock features with these text features did not unlock substantial predictive power with the MLP classifier, suggesting that this simple fusion approach may not effectively combine the information or that the noise levels are too high.

**3.4. Early Fusion: LSTM Stock Embeddings + Text Features (MLP)**

*   **Rationale:** To test if fusing a *learned representation* of stock history (LSTM embeddings) with text features yields better results. LSTM embeddings aim to capture complex temporal dynamics more effectively. (Using results from `fused_lstm_text_models_encoders_summary_lag1_hist1_2_3_5.csv` as it's a complete run).

   | Fused Model (LSTM Emb. + Text Type) | Accuracy (MDA) | ROC AUC | F1 (Macro) | Notes                                     |
   | :---------------------------------- | :------------- | :------ | :--------- | :---------------------------------------- |
   | LSTM Emb + FinBERT Sentiment        | 0.5355         | 0.5340  | 0.53       |                                           |
   | LSTM Emb + FinBERT Tone             | 0.5061         | 0.5159  | 0.49       |                                           |
   | LSTM Emb + FinBERT2                 | 0.5355         | 0.5461  | 0.53       |                                           |
   | LSTM Emb + RoBERTa Sentiment        | 0.5110         | 0.5021  | 0.51       |                                           |
   | LSTM Emb + BERT Base                | 0.5477         | 0.5323  | 0.55       | **Best overall performance observed**       |
   | LSTM Emb + ST_all-MiniLM-L6-v2      | 0.5012         | 0.5354  | 0.50       |                                           |

   *   **Observations:** This fusion strategy generally yielded the highest performance metrics. The combination of **`LSTM Emb + BERT Base` achieved the highest accuracy (0.5477) and F1-Macro (0.55)**. This modestly surpasses the stock-only XGBoost baseline (0.5386 accuracy) and other fusion/text-only approaches. It's noteworthy that `BERT Base`, which performed poorly in other contexts, showed utility here, possibly because the robust LSTM stock embeddings provided a strong contextual anchor that allowed the MLP to leverage some orthogonal signal from the general BERT Base text features.
   *   **Conclusion for LSTM Emb + Text Fusion:** Using pre-trained LSTM hidden states as a learned representation of stock history for fusion with text features appears to be the most promising approach among those tested. It suggests that abstracting stock market dynamics into an embedding before fusion can be more effective for the MLP than using raw historical features directly.

**4. Overall Conclusions and Future Work**

This comparative study investigated various models for daily stock direction prediction. The experiments consistently demonstrated the inherent difficulty of this task, with stock-only and text-only models generally performing near random chance (accuracies ~50-51%). Simple early fusion of raw historical stock features with diverse text features also failed to provide a significant predictive advantage.

The most promising results emerged from fusing learned LSTM stock embeddings (representing historical stock context) with text features, particularly when using `bert-base-uncased` as the text encoder. This combination achieved an accuracy of approximately 0.5477 and an F1-Macro of 0.55, modestly outperforming other configurations, including the strongest stock-only XGBoost baseline (0.5386 accuracy). This suggests that providing the final classifier with higher-level, learned representations from both modalities (LSTM for stock history, BERT-variants for text) is more beneficial than combining raw stock data with text features or using either modality in isolation for this daily prediction task.

However, the overall predictive accuracies remain low, indicating that predicting next-day stock direction based on daily aggregated news and historical stock data is exceptionally challenging. The primary limitation is likely the daily resolution of the data, which averages out any immediate intra-day market reactions to specific news events, thereby diluting the signal from news.

Future work should prioritize:
1.  **Acquiring Intra-day Data:** Transitioning to intra-day stock prices (e.g., 1-minute, 5-minute) and precisely timestamped news would allow for event-based analysis, focusing on short-term predictions immediately following specific news publications. This is crucial for more meaningfully assessing the impact of news.
2.  **Advanced Fusion Models:** Exploring attention mechanisms or multi-modal transformer architectures could better capture nuanced interactions between stock and text features.
3.  **Sophisticated Text Representations:** Fine-tuning language models specifically on financial news for tasks beyond sentiment, or leveraging larger language models (LLMs) for news summarization or feature extraction, might yield more potent text features.
4.  **Richer Stock Embeddings:** Experimenting with different LSTM architectures, training objectives, or other sequence models (e.g., Transformers) for generating stock history embeddings.

While the current models show limited practical predictive power for daily movements, the systematic comparison provides valuable insights into the relative performance of different feature types and fusion strategies, emphasizing the potential (albeit slight in this setup) of combining learned representations from multiple modalities.