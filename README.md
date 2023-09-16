# Forest-Fire-Prediction-with-Regression-and-Classification

## Content
- Abstract
- Introduction
- Data Transformation
- Variable Selection
- Regression Model
- Classification Model
- Conclusion
- Reference

# Abstract


本文所使用的資料來自 UCI Machine Learning Repository 中的 [Forest Fires Dataset](https://archive.ics.uci.edu/ml/datasets/forest+fires)。旨在透過空間、時間、火災天氣指數 (FWI) 和氣象變量 (M) 共 12 個參數，去預測火災造成的燃燒面積 (area)。由於燃燒面積的資料呈現嚴重的正偏，且存在離群值，因此需先進行轉換使其分佈更接近常態分佈後再進行分析。本文提出了不同於 P. Cortez *et al.*<sub>[5]</sub> 所使用的 Logarithmic transform，認為 Yeo-Johnson transform 才是最適合這筆資料集的轉換。本文所使用的監督式機器學習演算法包含迴歸 (Regression) 和分類 (Classification) 兩種，迴歸和分類之間的主要區別在於預測的輸出變量類型。 在迴歸中，輸出變量是連續的，而在分類中，輸出變量是離散的。迴歸分析方法有 Multiple Linear Regression (MR) 和 Support Vector Regression (SVR)，分類方法有 Logistic Regression 和 Support Vector Classification (SVC)。起初模型預測的效果不佳，我認為有 2 個原因。一是有過多的資料集中在 area = 0 (50% of data ≤ 0.52) ; 二是數據量過少 (僅 517 筆)。綜合以上兩點可以發現，當有過多的 area = 0 且 area 非 0 資料點過少時，會導致模型無法針對非 0 資料做學習，進而導致預測成果不佳。本文使用的解法是**上採樣** **(Upsample)**，在上採樣後，MR 模型的 $R^2$ 出現了 **3 到 10 倍**不等的增加，SVR 模型的 $R^2$ 也有 **10 倍以上**的增加。Regression 中以 SVR 的效果較佳，符合 P. Cortez *et al.*<sub>[5]</sub> 的結論，最佳的分組是**保留所有變數**，這點和論文不同， $R^2$ 達到 **0.73**，是相同條件下 MR 的 1.5 倍，但這兩個 Regression 模型卻均有一個共同的缺點就是對於 area = 0 的預測能力不佳，不過 Classification 可以有效地彌補這個缺點，Classification 中以 SVC 的結果較佳，最佳的分組為使用 **Stepwise selection** 選出的 9 個變數，Accuracy 和 Precision 均可以達到 **90%**，其他分組對於預測 area = 0 的能力也十分出色，Precision 幾乎都有接近 **90%** 的水準，這有效解決 Regression 難以對 area = 0 做出正確預測的問題，此外，SVC 模型對於 Small fire 和 Large fire 的預測 Accuracy 也達到了 70~80% 左右的水準。

# Introduction

Forest fire dataset 中共包含了四種變量，分別為空間、時間、火災天氣指數 (FWI) 和天氣變量 (M)，共 12 個參數 (Table. 1)，各參數的描述如 Fig. 1, 2。在這次的分析中，目標變數為 燃燒面積 “area”，燃燒面積隨空間分佈如 Fig. 3。然而，在對資料的描述 (Fig. 1-3) 和分布圖 (Fig. 2-8) 中可以觀察到，area 的平均值為 12.85，標準差為 63.7，大約有 75% 的資料集中在 0 到 6.6 公頃之間，而最大值則高達 1090.84 公頃。此外，area 的 skewness 和 kurtosis 分別為 12.85 和 194.14，且有 11 筆資料大於 100，可視為離群值。整體而言，資料的分佈呈現嚴重的正偏 (right-skewed)，違反了迴歸分析的基本假設中的常態性 (normality)。因此，我們需要對資料進行轉換才能進行迴歸分析。此外，也可以觀察到其他變數如 “FFMC”、”ISI”、”rain” 也存在離群值的情況。

## Forest Data Description

<img width="450" src="https://github.com/scfengv/Forest-Fire-Prediction-with-Regression-and-Classification/assets/123567363/b59880d7-34bc-496c-95a5-a109720048d9">

Fig. 1-1 [5]

<img width="800" alt="截圖 2023-09-16 23 27 07" src="https://github.com/scfengv/Forest-Fire-Prediction-with-Regression-and-Classification/assets/123567363/23a4fb4a-dff1-464b-b5ee-44f0402117c6">

Table. 1

## Quick look about dataset

<img width="300" alt="截圖 2023-03-29 14 56 45" src="https://github.com/scfengv/Forest-Fire-Prediction-with-Regression-and-Classification/assets/123567363/546d94cc-7210-42c9-b7b4-522721683ea9">
<img width="500" alt="截圖 2023-04-01 10 27 46" src="https://github.com/scfengv/Forest-Fire-Prediction-with-Regression-and-Classification/assets/123567363/ad5bbaa3-8572-436a-8c7d-819a1677d8d4">

Fig. 1-2 & Fig. 1-3

# Data Transformation

我參考了 A. Afifi *et al.* <sub>[1]</sub> 在 Annual Review of Public Health 發表的論文和網路文章 <sub>[2][3][4]</sub> 介紹了幾種用來處理極度不均衡的資料常見的轉換方法，我用了其中四個常見的轉換，分別為 1. Logarithmic ; 2. Square root ; 3. Cube root ; 4. Yeo-Johnson，並比較哪個才是最適用於這個資料集的轉換。評估標準包括 Skewness, Kurtosis 和 QQ plot。

Skewness 是衡量分佈的**對稱性**。 如果分佈不對稱，則稱該分佈是偏斜 (skewed) 的，這意味著分佈的左側和右側具有不同的形狀。 偏度可以是正數、負數或零。 正偏度意味著分佈的右尾很長 (目標變數 area 的分佈)，而負偏度意味著分佈的左尾很長， 零偏度表示分佈是對稱的。

Kurtosis 是分佈的**峰度**或平坦度的度量。 峰度高的分佈有 Sharp peak 和 Heavy tails，而峰度低的分佈有 Flat peak 和 Light tails。 峰度也可以是正數、負數或零。 正峰度表示尖峰，負峰度表示平峰。

QQ plot 是 quantile-quantile plot 的簡稱，一種用於比較兩個機率分佈的圖形。 它根據常態分佈的分位數繪製樣本數據的分位數。 如果樣本數據呈常態分佈，則 QQ 圖上的點將位於一條直線上。 如果樣本數據存在Skewed 或 Heavy tail 現象，QQ plot 上的點會偏離直線，表明樣本分佈不正常。 QQ plot 是檢查統計分析中**常態性**假設的常用工具。

為什麼要做轉換呢？就像 M. Bland 在他的文章中所說的 “*Even if a transformation does not produce a really good fit to the Normal distribution, it may still make the data much more amenable to analysis* ”<sub>[4]</sub> ，關於這個概念我想用 area 的轉換來說明 (Fig. 4)。比較左右兩張圖可知，經過 Log 轉換後 (Fig. 4-2)，用來量化常態性的指標 Skewness 和 Kurtosis 都有顯著的下降，QQ plot 也更接近一條直線，說明了這筆資料在經過轉換後會越來越接近常態分佈。

<img width="450" src="https://github.com/scfengv/Forest-Fire-Prediction-with-Regression-and-Classification/assets/123567363/c6aa7ab4-9262-4839-ad48-1c64148a75d0">


<img width="450" src="https://github.com/scfengv/Forest-Fire-Prediction-with-Regression-and-Classification/assets/123567363/e3e33485-dfab-450a-b375-f4bda7d8e7c4">

Fig. 4-1

$$
Skewness:3.50/2.45 \ \ \ \ \ Kurtosis:13.88/6.34
$$

<img width="450" src="https://github.com/scfengv/Forest-Fire-Prediction-with-Regression-and-Classification/assets/123567363/623d3a15-8f53-49dc-b686-b8f8b29b8501">


<img width="450" src="https://github.com/scfengv/Forest-Fire-Prediction-with-Regression-and-Classification/assets/123567363/e491ff15-d956-4997-b81c-d6d18a52bbed">

Fig. 4-2

$$
Skewness:1.21/0.43 \ \ \ \ \ Kurtosis:0.95/-0.60
$$

## Transformation method comparison

**With area = 0**

<img width="727" alt="截圖 2023-09-16 23 49 15" src="https://github.com/scfengv/Forest-Fire-Prediction-with-Regression-and-Classification/assets/123567363/bdf42574-2111-4cb1-9e29-05fcb0bcfab6">

**Without area = 0**

<img width="731" alt="截圖 2023-09-16 23 56 40" src="https://github.com/scfengv/Forest-Fire-Prediction-with-Regression-and-Classification/assets/123567363/b9ecc055-a7e6-42dd-9ce0-697b5c4ceefb">


從兩張圖的比較可以發現，經過各種轉換後，Skewness 和 Kurtosis 相較於原始資料都有明顯下降，也有使 QQ plot 的離群值減少不少。但需要注意的是，不論哪種轉換方式，仍然受到了過多的 area = 0 的資料的影響，這些資料約佔整體資料的一半 (244 筆)。因此在進一步分析時，如果將焦點放在非 0 的 area 資料上，從轉換後的數據可以明顯地看出經過轉換後的資料都呈現出很接近常態的分佈，尤其以 Yeo-Johnson transform 的效果最為顯著。因此，在這筆資料中，我認為使用 Yeo-Johnson transform 是最適合的轉換方式。

# Variables Selection

在變數選擇上，除了引用 P. Cortez *et al.* <sub>[5]</sub> 將資料分為 STFWI, STM, FWI, M 四組外 (Table. 2)，我用了 Wrapper method 中的 Sequential Forward / Backward / Forward floating / Backward floating Selection 共四種方法去找出最重要且適合的變數 <sub>[6][7][12]</sub>。

四條曲線都在約 15 個變數左右時斜率變化趨近平緩，故先挑出各組的影響力前 15 ，其中四組共同擁有的 9 個變數為 [ ”X”, “FFMC”, “DMC”, “ISI”, “temp”, “wind”, “month_dec”, “month_nov”, “day_fri” ]，故將這 9 個變數也列為一組和論文中的其他四組做比較。

| Feature selection | Variables |
| --- | --- |
| STFWI | Spatial + Temporal + FWI |
| STM | Spatial + Temporal + Weather  |
| FWI | FWI |
| M | Weather |

Table. 2

<img width="1200" alt="截圖 2023-04-01 22 52 31" src="https://github.com/scfengv/Forest-Fire-Prediction-with-Regression-and-Classification/assets/123567363/fb52e839-e640-4dee-9584-117b8aefcb5e">

# Regression Model

為了要預測火災的燃燒面積，我對資料集進行了監督式學習中的迴歸預測分析，包含 Multiple Linear Regression (MR) 和 Support Vector Regression (SVR)

Multiple Linear Regression 是 Ordinary Least-Square Regression 的延伸，使用多個自變量來預測應變量的結果。 MR 的目標是最小化預測值和實際值之間的平方差來模擬解釋自變量和應變量之間的線性關係 <sub>[18]</sub>，但 SVR 是求係數的 L2-norm 的最小化。Support Vector 的概念簡單來說就是將資料從低維度空間中投影至高維度空間，使原本在低維度無法進行切割的資料，在高維度時能找到超平面來分開樣本 <sub>[9][10]</sub>，SVR 中的誤差項通過設置誤差上限 $\epsilon$ 來限制，使得預測值和實際值之間的絕對差值小於或等於 $\epsilon$，通過調整 $\epsilon$ 的值，我們可以控制 SVR 模型的準確性。較小的 $\epsilon$ 值意味著模型對誤差的容忍度更嚴格，這會導致 Overfitting，而較大的 $\epsilon$  值會導致 Underfitting。簡單來說，SVR 使我們能夠靈活地定義我們的模型可以接受多少誤差，並找到合適的直線或更高維度的超平面來擬合數據 <sub>[19][20]</sub>
<br>

$$
L_{2}=(\sum_{i=1}^n \| x \| _{i}^2)^{1/2}
$$

## Multiple Linear Regression

### Original Data

<img width="700" alt="截圖 2023-09-17 00 36 09" src="https://github.com/scfengv/Forest-Fire-Prediction-with-Regression-and-Classification/assets/123567363/6159b316-eb6c-4b6a-b8ce-54c1a9967c4b">

<img width="700" src="https://github.com/scfengv/Forest-Fire-Prediction-with-Regression-and-Classification/assets/123567363/cbcda2d5-38d9-4509-810e-cab2df8ccbc0">


### Upsample Data
<img width="700" alt="截圖 2023-09-17 00 36 41" src="https://github.com/scfengv/Forest-Fire-Prediction-with-Regression-and-Classification/assets/123567363/7caefc10-6acf-4f8b-93fc-56083e32faf4">

<img width="700" src="https://github.com/scfengv/Forest-Fire-Prediction-with-Regression-and-Classification/assets/123567363/952f8c24-786c-4e0c-9564-0eb834d04ed1">

### Upsample Data Without area = 0
<img width="700" alt="截圖 2023-09-17 00 37 01" src="https://github.com/scfengv/Forest-Fire-Prediction-with-Regression-and-Classification/assets/123567363/c7124344-b713-4a30-8f1c-345589a31bb1">

<img width="700" src="https://github.com/scfengv/Forest-Fire-Prediction-with-Regression-and-Classification/assets/123567363/664015e7-89a1-491a-9267-6f6d6e1ea690">

---

## Support Vector Regression

### Original Data

<img width="700" alt="截圖 2023-09-17 00 47 38" src="https://github.com/scfengv/Forest-Fire-Prediction-with-Regression-and-Classification/assets/123567363/92948329-0ad7-4503-87a2-49890313e6e5">

<img width="700" src="https://github.com/scfengv/Forest-Fire-Prediction-with-Regression-and-Classification/assets/123567363/ab52b51e-9071-4793-bfd4-4d7f541527cc">

### Upsample Data

<img width="700" alt="截圖 2023-09-17 00 48 00" src="https://github.com/scfengv/Forest-Fire-Prediction-with-Regression-and-Classification/assets/123567363/90a3da60-b485-464c-84cb-122115888ddc">

<img width="700" src="https://github.com/scfengv/Forest-Fire-Prediction-with-Regression-and-Classification/assets/123567363/8df31b4a-ef3c-4b3f-9771-accb4a77f035">

### Upsample Data Without area = 0

<img width="700" alt="截圖 2023-09-17 00 48 20" src="https://github.com/scfengv/Forest-Fire-Prediction-with-Regression-and-Classification/assets/123567363/8e53df87-1cda-4e02-99c4-2ed7dc372518">

<img width="700" src="https://github.com/scfengv/Forest-Fire-Prediction-with-Regression-and-Classification/assets/123567363/8f23c189-3914-4a22-ad14-b8cd7a2d8c43">

觀察原始資料集進行的 MR 和 SVR 預測表現並不理想。這可能是由於資料過度集中在 0 附近 (50% 的資料 ≤ 0.52)，而另外一半資料的差異範圍卻很大，同時，由於資料集的大小只有 517 筆，在將近一半的資料為 0 且 非 0 資料很少又差異極大的情況下，導致模型難以針對非 0 資料學習並做出正確預測，在一個 Imbalance 的 dataset 中，多數類 (area = 0) 的數據點數量遠遠高於少數類的數量。這可能會導致模型性能的偏差，因為模型傾向於預測多數類比少數類更頻繁。

為了解決這個問題，我參考了處理不平衡資料的常見方法，即使用上採樣 (Data Upsample)，上採樣是透過增加少數類別的樣本數，使得少數類別的樣本數與多數類別相當，進而提高模型預測能力 。我所使用的方法是 Python 中的 random.sample() function <sub>[11]</sub>，在各個月份隨機生成 200 組相近的數據點，使每個月份的數據點數量相近，稱其為 **Upsample by month**。使用這個方法後，可以看到確實對於每個群體的 $R^2$ 都提升了 3 到 10 倍不等，但是模型仍然受到過多 area = 0 的資料點的影響，導致無法正確預測，可以從結果看到，test set 中 area = 0 的資料點幾乎都不在迴歸線上。

為此，我嘗試使用不包含 area = 0 的資料進行迴歸，但結果並不如預期。這可能是因為某些月份 (例如 11 月) 的數據結果都是 area = 0，如果排除這些資料點，那麼這些月份的資料就會消失，導致模型產生更大的誤差。

經參考 P. Cortez *et al.* <sub>[5]</sub> 的研究結果，得出 Support Vector Machine (SVM) 可能是最適合這筆資料的 Data mining model。同 MR 分別做了 Upsample 和 Upsample Without area = 0 兩種處理方式，而確實也相較於 MR 得到的更好的預測結果。最佳的分組是保留所有變數，有 / 無包含 area = 0 的 $R^2$ 分別達到 0.61 和 0.73，是 MR 在同樣條件下的 1.5 倍，其他分組如 STFWI 和 STM 的 $R^2$ 也達到 0.6 以上，Feature selection 選出的 9 個變數 $R^2$ 也將近 0.6，較原始資料集的預測高上非常多。

# Classification Model

為了解決無法準確預測 area = 0 的情況，我嘗試將預測從迴歸模型 (Regression) 轉成分類問題 (Classification)，透過將 area 分成 **no fire** (area = 0) / **small fire** (0 < area < 6) / **large fire** (6 < area) 三組，去預測潛在的火勢大小以決定需要調派的資源，可能會比準確預測一個燃燒的面積來得有意義。Upsample by month 前 / 後三組的比例如下 (Fig. 12-1, 12-2)，可以看到這個方法大致以一樣的比例去做 Upsample，但仍會有一點微小的差異。

使用的模型有 Logistic Regression 和 Support Vector Classification。Logistic Regression 根據給定的自變量數據集估計事件發生的概率。 由於結果是機率，因此應變量介於 0 和 1 之間。Logistic Regression 是一個平滑的曲線 (Fig. 11-1) $Z=\beta_0+\beta_1*x_1+......+\beta_n*x_n$，其中會通過 Maximum Likelihood Function (MLE) 多次迭代 (iteration) 試圖最大化該函數以找到最佳參數 $\beta$ ，一旦找到最佳係數，就可以計算每個變量的條件機率，並將它們加在一起以產生預測概論。簡單來說，當 $Z$ 越大時，判斷成 A 類的機率越大，反之判斷成A類的機率越小 <sub>[22][23]</sub>

<img width="700" src="http://rasbt.github.io/mlxtend/user_guide/classifier/LogisticRegression_files/logistic_function.png">

Fig. 11-1 Logistic Sigmoid Function [21]

Upsample 的方法先經過前面提到的 Upsample by month 後再透過 Imbalanced learn 中的 SMOTE 調整，sampling strategy 均令為 “Auto”。

SMOTE 的採樣模式是通過在現有的少數類別樣本之間進行插值，為少數類別生成合成樣本。具體來說，它隨機選擇一個少數群體的樣本，並在特徵空間 (feature space) 中找到它的 k 個最近的鄰居 (k nearest neighbors)，然後隨機選擇這些鄰居中的一個，並在連接原始少數群體樣本和所選鄰居的線段上隨機選擇一個點，生成一個新的合成樣本 <sub>[14][15]</sub>。

量化模型的指標為 Accuracy, Precision, F1-score, Confusion matrix & ROC curve 

$$
Accuracy=\frac{TP+TN}{ALL}\ \ \ \ Precision=\frac{TP}{TP+FP}\ \ \ \ Recall=\frac{TP}{TP+FN}\ \ \ \ F1=2* \frac{Precision*Recall}{Precision+Recall}
$$

本篇由於是將資料分為三組，所以 Confusion matrix 會有別於傳統的 Positive & Negative，會以三組 Positive & Negative 的方式呈現，端看當下在預測的是哪一組

|  | True No fire | True Small fire | True Large fire |
| --- | --- | --- | --- |
| Predicted Large fire | False | False | True Large |
| Predicted Small fire | False | True Small | False |
| Predicted No fire | True No | False | False |

Accuracy: 正確分類率，指的是實際上為正且被預測為正 (TP) + 實際上為負且被預測為負 (TN) 的機率

Precision: 精準度，定義為所有被預測為正確 (TP + FP) 的樣本中，有多少比例實際上是正確的 (TP)

Recall: 召回率，定義為實際上是正確的樣本中 (TP + FN)，有多少比例是被預測為正確的 (TP)

F1 score: 綜合評估 Precision & Recall

ROC curve 的組成 x 軸為 False Positive rate, y 軸為 True Positive rate (Fig. 11-2)，最理想的模型 Area under curve (AUC) = 1 (Green line) 會通過圖的左上角，表示 TPR = 1, FPR = 0，不管閥值 (decision threshold) 為何都可以 100% 預測 ; 當 AUC = 0.5 (Red line) 時，曲線幾乎等於對角線，模型沒有任何鑑別度，因為不管你正樣本分對比例多高，永遠都會有同等比例的負樣本被錯判。簡單來說，當 ROC curve 越接近左上角，表示模型越準確，可以用曲線下面積 AUC 來衡量  <sub>[16]</sub>

<img width="700" src="https://github.com/scfengv/Forest-Fire-Prediction-with-Regression-and-Classification/assets/123567363/abb95b9c-3829-433e-816e-a5548af867d2">

Fig. 11-2 [17]

<img width="475" alt="截圖 2023-04-02 17 12 27" src="https://github.com/scfengv/Forest-Fire-Prediction-with-Regression-and-Classification/assets/123567363/0a1bdb95-5c5d-4bd6-a49b-e190c621ded4">
<img width="475" alt="截圖 2023-04-02 17 12 57" src="https://github.com/scfengv/Forest-Fire-Prediction-with-Regression-and-Classification/assets/123567363/50aa2da4-b6f7-4514-ba59-dd003869c7e1">

Fig. 12-1 (Upsample 前) & Fig. 12-2 (Upsample 後)

## Difference between different sampling strategy in imbalanced learn

- Auto: 對所有類別進行重新採樣，使其樣本數與樣本數中位數的類別相同，當希望算法自動平衡類別時，此策略很有用。
- Not minority: 對除少數類別之外的所有類別 (No, Large fire) 進行重新採樣。 該算法將根據需要盡可能刪除多的多數類別樣本。
- Minority: 對少數類別 (Small Fire) 進行重新採樣，使其與多數類別 (No fire) 具有相同數量的樣本。該策略將少數類別的目標比例設置為等於1，意味著算法將根據需要盡可能生成多的少數類樣本以達到該比例。
- Not majority: 對除了多數類別以外的所有類別 (Small, Large fire) 進行重新採樣。意味著 dataset 將是平衡的。 該算法將根據需要盡可能生成多的少數類樣本以實現這種平衡。

在 Sampling strategy 的選擇上，不僅僅是希望能補上 Regression 模型難以預測的 No fire 部分，更希望能對 Small fire 和 Large fire 也有很好的預測能力，故在 Strategy 的選擇上選了均衡的 “auto”，而非偏向 Large fire 的 “Not minority” 或偏向 Small fire 的 “Minority”，但這也可以依想預測的類別做調整，並沒有一定的答案。

---

## Logistic Regression

Red curve: No fire ; Green curve: Small fire ; Blue curve: Large fire

<img width="800" alt="截圖 2023-09-17 01 26 36" src="https://github.com/scfengv/Forest-Fire-Prediction-with-Regression-and-Classification/assets/123567363/851aed75-3e4d-4838-b9b0-685996524d93">

<img width="600" src="https://github.com/scfengv/Forest-Fire-Prediction-with-Regression-and-Classification/assets/123567363/ff744098-06aa-4712-863d-2eb8777e13ad">

Fig. 14

---

## Support Vector Classification

<img width="800" alt="截圖 2023-09-17 01 25 01" src="https://github.com/scfengv/Forest-Fire-Prediction-with-Regression-and-Classification/assets/123567363/c50f417a-6b9e-4bd1-8283-68bdf9ceda36">

<img width="400" src="https://github.com/scfengv/Forest-Fire-Prediction-with-Regression-and-Classification/assets/123567363/7d22dc86-12ba-40c7-a3db-700959b2e553">

Fig. 15

根據 Classification 的分析結果 (Fig. 14, 15)，可以看出 **SVC** 的表現明顯優於 Logistic Regression。Logistic Regression 中最好的模型是**保留所有變數** ，AUC 為 **0.83** 或更高，但對於 FWI 和 M 兩組而言，AUC 僅只有 0.6 左右，幾乎等於隨機猜測。此外， 針對綜合評估 Precision 和 Recall 的 F1 score，Logistic Regression 普遍不及 SVC 模型。值得注意的是，SVC 模型的 F1 score 除了 FWI 那組外均達到 **0.87 以上**，且幾乎對 No fire 的識別率都達到 **90%** 以上。在 SVC 的所有模型中，Stepwise selection 的 9 個變量表現最佳，F1 score 為 **0.90**。該模型可以成功識別 **96.25%** 的 No fire，對 Small fire 和 Large fire 分別也有將近 8 成和 8 成以上的準確率。它不僅成功彌補了 Regression 模型難以預測的 area = 0 的缺陷，同時也展現出對其他類別的出色識別能力。

# Conclusion

回歸到問題本身，Classification 中的 SVC 模型可以準確識別 No fire 也就是 area = 0 的部分。若要預測火災發生面積，可以採用**保留所有變數且不包含 area = 0 的  SVR 模型**，儘管這部分仍然需要進一步優化 ; 對於僅需要大致預測火災規模的需求，**SVC 中的 Stepwise selection 模型**已經可以達到接近 90% 的精準度，而其他模型也均達到了 80% 以上的水準。若需要精準的預測 Small fire, Large fire 的火災發生面積，則可以採用 Classification 後再在各類別內做 Regression，這是一個可以進一步延伸的方向。

綜觀以上，由於資料集的特性和分佈狀態，使用 Regression 模型進行精確的預測是非常困難的，然而，仍可以透過排除 area = 0 的資料，並利用上採樣的方式提升資料量以供模型來做學習，從而提升預測非 0 燃燒面積的精確度。至於預測 area = 0 的部分，可以巧妙的利用 Classification 將 area = 0 視為一個類別來進行預測，並收到非常好的效果。

# Reference

[1] A.A. Afifi, J.B. Kotlerman, S.L. Ettner, M. Cowan. Methods for improving regression analysis for skewed continuous or counted responses. Annual Review of Public Health, 28 (2007), pp. 95-111, [10.1146/annurev.publhealth.28.082206.094100](https://doi.org/10.1146/annurev.publhealth.28.082206.094100)

[2] 資料分布與離群值處理 https://ithelp.ithome.com.tw/articles/10278000

[3] Feature Engineering -- 3. Variable transformation https://ithelp.ithome.com.tw/articles/10235219

[4] M. Bland. Transformation. University of York https://www-users.york.ac.uk/~mb55/msc/clinbio/week5/transfm.htm

[5] Cortez, P., & Morais, A. D. J. R. (2007). A data mining approach to predict forest fires using meteorological data

[6] SequentialFeatureSelector: The popular forward and backward feature selection approaches (including floating variants). http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/

[7] A comprehensive guide to Feature Selection using Wrapper methods in Python https://www.analyticsvidhya.com/blog/2020/10/a-comprehensive-guide-to-feature-selection-using-wrapper-methods-in-python/

[8] 7 Techniques to Handle Imbalanced Data https://www.kdnuggets.com/2017/06/7-techniques-handle-imbalanced-data.html

[9] 支持向量機(SVM, SVC) https://ithelp.ithome.com.tw/m/articles/10299494

[10] [資料分析&機器學習] 第3.4講：支援向量機(Support Vector Machine)介紹 [https://medium.com/jameslearningnote/資料分析-機器學習-第3-4講-支援向量機-support-vector-machine-介紹-9c6c6925856b](https://medium.com/jameslearningnote/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC3-4%E8%AC%9B-%E6%94%AF%E6%8F%B4%E5%90%91%E9%87%8F%E6%A9%9F-support-vector-machine-%E4%BB%8B%E7%B4%B9-9c6c6925856b)

[11] Python | random.sample() function https://www.geeksforgeeks.org/python-random-sample-function/

[12] Five ways to choose the right variables to build the right models  https://medium.com/analytics-vidhya/five-ways-to-choose-the-right-variables-to-build-the-right-models-4827f86f1583

[13] RandomOverSampler API https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html

[14] 5 SMOTE Techniques for Oversampling your Imbalance Data. https://towardsdatascience.com/5-smote-techniques-for-oversampling-your-imbalance-data-b8155bdbe2b5

[15] SMOTE API https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html

[16] SMOTE + ENN : 解決數據不平衡建模的採樣方法 [https://medium.com/數學-人工智慧與蟒蛇/smote-enn-解決數據不平衡建模的採樣方法-cdb6324b711e](https://medium.com/%E6%95%B8%E5%AD%B8-%E4%BA%BA%E5%B7%A5%E6%99%BA%E6%85%A7%E8%88%87%E8%9F%92%E8%9B%87/smote-enn-%E8%A7%A3%E6%B1%BA%E6%95%B8%E6%93%9A%E4%B8%8D%E5%B9%B3%E8%A1%A1%E5%BB%BA%E6%A8%A1%E7%9A%84%E6%8E%A1%E6%A8%A3%E6%96%B9%E6%B3%95-cdb6324b711e)

[17] AUC (Area under the ROC Curve) https://machine-learning.paperspace.com/wiki/auc-area-under-the-roc-curve

[18] Multiple Linear Regression (MLR) Definition, Formula, and Example https://www.investopedia.com/terms/m/mlr.asp

[19] An Introduction to Support Vector Regression https://towardsdatascience.com/an-introduction-to-support-vector-regression-svr-a3ebc1672c2

[20] R筆記 – (14)Support Vector Machine/Regression(支持向量機SVM) https://rpubs.com/skydome20/R-Note14-SVM-SVR

[21] LogisticRegression: A binary classifier http://rasbt.github.io/mlxtend/user_guide/classifier/LogisticRegression/

[22] What is logistic regression? https://www.ibm.com/topics/logistic-regression

[23] [資料分析&機器學習] 第3.3講：線性分類-邏輯斯回歸(Logistic Regression) 介紹 [https://medium.com/jameslearningnote/資料分析-機器學習-第3-3講-線性分類-邏輯斯回歸-logistic-regression-介紹-a1a5f47017e5](https://medium.com/jameslearningnote/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC3-3%E8%AC%9B-%E7%B7%9A%E6%80%A7%E5%88%86%E9%A1%9E-%E9%82%8F%E8%BC%AF%E6%96%AF%E5%9B%9E%E6%AD%B8-logistic-regression-%E4%BB%8B%E7%B4%B9-a1a5f47017e5)
