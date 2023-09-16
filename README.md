# Forest-Fire-Prediction-with-Regression-and-Classification
# Abstract


本文所使用的資料來自 UCI Machine Learning Repository 中的 Forest Fires Dataset (https://archive.ics.uci.edu/ml/datasets/forest+fires)。旨在透過空間、時間、火災天氣指數 (FWI) 和氣象變量 (M) 共 12 個參數，去預測火災造成的燃燒面積 (area)。由於燃燒面積的資料呈現嚴重的正偏，且存在離群值，因此需先進行轉換使其分佈更接近常態分佈後再進行分析。本文提出了不同於 P. Cortez *et al. $_{[5]}$* 所使用的 Logarithmic transform，認為 Yeo-Johnson transform 才是最適合這筆資料集的轉換。本文所使用的監督式機器學習演算法包含迴歸 (Regression) 和分類 (Classification) 兩種，迴歸和分類之間的主要區別在於預測的輸出變量類型。 在迴歸中，輸出變量是連續的，而在分類中，輸出變量是離散的。迴歸分析方法有 Multiple Linear Regression (MR) 和 Support Vector Regression (SVR)，分類方法有 Logistic Regression 和 Support Vector Classification (SVC)。起初模型預測的效果不佳，我認為有 2 個原因。一是有過多的資料集中在 area = 0 (50% of data ≤ 0.52) ; 二是數據量過少 (僅 517 筆)。綜合以上兩點可以發現，當有過多的 area = 0 且 area 非 0 資料點過少時，會導致模型無法針對非 0 資料做學習，進而導致預測成果不佳。本文使用的解法是**上採樣** **(Upsample)**，在上採樣後，MR 模型的 $R^2$ 出現了 **3 到 10 倍**不等的增加，SVR 模型的 $R^2$ 也有 **10 倍以上**的增加。Regression 中以 SVR 的效果較佳，符合 P. Cortez *et al. $_{[5]}$* 的結論，最佳的分組是**保留所有變數**，這點和論文不同，$R^2$ 達到 **0.73**，是相同條件下 MR 的 1.5 倍，但這兩個 Regression 模型卻均有一個共同的缺點就是對於 area = 0 的預測能力不佳，不過 Classification 可以有效地彌補這個缺點，Classification 中以 SVC 的結果較佳，最佳的分組為使用 **Stepwise selection** 選出的 9 個變數，Accuracy 和 Precision 均可以達到 **90%**，其他分組對於預測 area = 0 的能力也十分出色，Precision 幾乎都有接近 **90%** 的水準，這有效解決 Regression 難以對 area = 0 做出正確預測的問題，此外，SVC 模型對於 Small fire 和 Large fire 的預測 Accuracy 也達到了 70~80% 左右的水準。

# 1. Introduction

Forest fire dataset 中共包含了四種變量，分別為空間、時間、火災天氣指數 (FWI) 和天氣變量 (M)，共 12 個參數 (Table. 1)，各參數的描述如 Fig. 1, 2。在這次的分析中，目標變數為 燃燒面積 “area”，燃燒面積隨空間分佈如 Fig. 3。然而，在對資料的描述 (Fig. 1-3) 和分布圖 (Fig. 2-8) 中可以觀察到，area 的平均值為 12.85，標準差為 63.7，大約有 75% 的資料集中在 0 到 6.6 公頃之間，而最大值則高達 1090.84 公頃。此外，area 的 skewness 和 kurtosis 分別為 12.85 和 194.14，且有 11 筆資料大於 100，可視為離群值。整體而言，資料的分佈呈現嚴重的正偏 (right-skewed)，違反了迴歸分析的基本假設中的常態性 (normality)。因此，我們需要對資料進行轉換才能進行迴歸分析。此外，也可以觀察到其他變數如 “FFMC”、”ISI”、”rain” 也存在離群值的情況。

## Forest Data Description

![Fig. 1-1 [5]](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/81a551d7-89ff-484d-a767-c861c0129e0c/Untitled.png)

Fig. 1-1 [5]

| Index | Variables | Description | Influence | Time lag | Weather |
| --- | --- | --- | --- | --- | --- |
| Spatial | X | x axis coordinate (1 ~ 9) |  |  |  |
|  | Y | y axis coordinate (1 ~ 9) |  |  |  |
| Temporal | Month | Month of the year (Jan. to Dec.) |  |  |  |
|  | Day | Day of the week (Mon. to Sun.) |  |  |  |
| FWI variables | FFMC | The moisture content surface litter | Ignition and Fire spread | 16 hr | Rain
Relative Humidity
Temperature
Wind |
|  | DMC | The moisture content of shallow organic layer | Fire intensity | 12 days | Rain
Relative Humidity
Temperature |
|  | DC | The moisture content of deep organic layer | Fire intensity | 52 days | Rain
Temperature |
|  | ISI | Fire velocity spread | Fire velocity |  |  |
| Weather condition | temp | Outside temperature |  |  |  |
|  | RH | Outside relative humidity |  |  |  |
|  | wind | Outside wind speed |  |  |  |
|  | rain | Outside rain |  |  |  |

Table. 1

## Quick look about dataset

![Fig. 1-2](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/89419a3e-4a28-481b-9ffb-a5d8c4fdc67b/%E6%88%AA%E5%9C%96_2023-03-29_14.56.45.png)

Fig. 1-2

![Fig. 1-3](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/df153a81-7417-4776-8553-f7c8656adfb5/%E6%88%AA%E5%9C%96_2023-04-01_10.27.46.png)

Fig. 1-3
![FFMC](https://github.com/scfengv/Forest-Fire-Prediction-with-Regression-and-Classification/assets/123567363/4287e44d-3daa-4833-acda-b3d6edb004ac)

![Fig. 2-1](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a3764e54-a139-4f04-bbb4-6484ed999514/FFMC.png)

Fig. 2-1

![Fig. 2-2](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/751dd890-70f5-4b89-90fd-44179679cddc/DMC.png)

Fig. 2-2

![Fig. 2-3](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b72e6345-8779-483c-a091-ecc144ccac8c/DC.png)

Fig. 2-3

![Fig. 2-4](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4b369430-7ef4-48ff-8d7d-6676b4844d13/ISI.png)

Fig. 2-4

![Fig. 2-5](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c94f8eff-c60e-4e65-b679-4eb1740387e7/RH.png)

Fig. 2-5

![Fig. 2-6](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/83a1833d-e0e9-452e-a300-c8223b27a436/wind.png)

Fig. 2-6

![Fig. 2-7](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3e170e77-ed87-43a8-92da-381a7c89df53/rain.png)

Fig. 2-7

![Fig. 2-8](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/03660943-f41b-4387-a0a8-534e9c06ddd0/area.png)

Fig. 2-8

![Fig. 3](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2c6d270e-9975-4936-9aec-1fff214051e4/newplot.png)

Fig. 3

# 2. Transformation

我參考了 A. Afifi *et al.* $_{[1]}$ 在 Annual Review of Public Health 發表的論文和網路文章 $_{[2][3][4]}$ 介紹了幾種用來處理極度不均衡的資料常見的轉換方法，我用了其中四個常見的轉換，分別為 1. Logarithmic ; 2. Square root ; 3. Cube root ; 4. Yeo-Johnson，並比較哪個才是最適用於這個資料集的轉換。評估標準包括 Skewness, Kurtosis 和 QQ plot。

Skewness 是衡量分佈的**對稱性**。 如果分佈不對稱，則稱該分佈是偏斜 (skewed) 的，這意味著分佈的左側和右側具有不同的形狀。 偏度可以是正數、負數或零。 正偏度意味著分佈的右尾很長 (目標變數 area 的分佈)，而負偏度意味著分佈的左尾很長， 零偏度表示分佈是對稱的。

Kurtosis 是分佈的**峰度**或平坦度的度量。 峰度高的分佈有 Sharp peak 和 Heavy tails，而峰度低的分佈有 Flat peak 和 Light tails。 峰度也可以是正數、負數或零。 正峰度表示尖峰，負峰度表示平峰。

QQ plot 是 quantile-quantile plot 的簡稱，一種用於比較兩個機率分佈的圖形。 它根據常態分佈的分位數繪製樣本數據的分位數。 如果樣本數據呈常態分佈，則 QQ 圖上的點將位於一條直線上。 如果樣本數據存在Skewed 或 Heavy tail 現象，QQ plot 上的點會偏離直線，表明樣本分佈不正常。 QQ plot 是檢查統計分析中**常態性**假設的常用工具。

為什麼要做轉換呢？就像 M. Bland 在他的文章中所說的 “*Even if a transformation does not produce a really good fit to the Normal distribution, it may still make the data much more amenable to analysis* ”$_{[4]}$ ，關於這個概念我想用 area 的轉換來說明 (Fig. 4)。比較左右兩張圖可知，經過 Log 轉換後 (Fig. 4-2)，用來量化常態性的指標 Skewness 和 Kurtosis 都有顯著的下降，QQ plot 也更接近一條直線，說明了這筆資料在經過轉換後會越來越接近常態分佈。

![output.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b56f84b6-8e07-48fc-a34d-d8c792acb2b6/output.png)

![Fig. 4-1](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/dc126322-8afb-4a19-8167-70d87c1e9c43/output.png)

Fig. 4-1

$$
Skewness:3.50/2.45
\\
Kurtosis:13.88/6.34
$$

![output.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8573a5d9-8935-49fe-8fbb-0b36a577b95c/output.png)

![Fig. 4-2](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9f6e3772-0b36-47c1-8ba7-a6c481a3b7ef/output.png)

Fig. 4-2

$$
Skewness:1.21/0.43
\\
Kurtosis:0.95/-0.60
$$

## Data transformation

### Original data

**With area = 0**

![Fig. 5-1](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e904f7fe-913a-4825-bc80-98bc9c3dc401/output.png)

Fig. 5-1

$$
Skewness:12.85
\\
Kurtosis:194.14
$$

**Without area = 0**

![Fig. 6-1](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7a49b234-5bc9-45d5-a339-1b5ce96fb96a/output.png)

Fig. 6-1

$$
Skewness:9.45
\\
Kurtosis:103.74
$$

### Logarithmic

**With area = 0**

![Fig. 5-2](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/47d77387-af11-41ce-bb2e-d3627305cd94/output.png)

Fig. 5-2

$$
Skewness:1.22
\\
Kurtosis:0.95
$$

**Without area = 0**

![Fig. 6-2](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1b0f20a4-84d5-40cd-a026-607a9eb3b836/output.png)

Fig. 6-2

$$
Skewness:0.30
\\
Kurtosis:0.30
$$

### Square root

**With area = 0**

![Fig. 5-3](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4df5033a-0539-49a6-b72f-17f1ea092258/output.png)

Fig. 5-3

$$
Skewness:4.34
\\
Kurtosis:30.75
$$

**Without area = 0**

![Fig. 6-3](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/cee73482-b3b3-4143-b30e-ca4829256dc5/output.png)

Fig. 6-3

$$
Skewness:4.10
\\
Kurtosis:25.13
$$

### Cube root

**With area = 0**

![Fig. 5-4](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/608a8ddd-6316-4696-8016-987fa3c2361b/output.png)

Fig. 5-4

$$
Skewness:1.83
\\
Kurtosis:5.87
$$

**Without area = 0**

![Fig. 6-4](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fdd7629c-56a4-407f-86d3-1324f0968df5/output.png)

Fig. 6-4

$$
Skewness:2.45
\\
Kurtosis:9.84
$$

### Yeo-Johnson

**With area = 0**

![Fig. 5-5](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/77f9a523-d609-4086-ae27-0d0800747d41/output.png)

Fig. 5-5

$$
Skewness:0.40
\\
Kurtosis:-1.48
$$

**Without area = 0**

![Fig. 6-5](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a17d29e6-3624-4364-807b-7fb075f717b9/output.png)

Fig. 6-5

$$
Skewness:0.08
\\
Kurtosis:-0.67
$$

從 Fig. 5-1 ~ 5-5 可以發現，經過各種轉換後，Skewness 和 Kurtosis 相較於原始資料都有明顯下降，也有使 QQ plot 的離群值減少不少。但需要注意的是，不論哪種轉換方式，仍然受到了過多的 area = 0 的資料的影響，這些資料約佔整體資料的一半 (244 筆)。因此在進一步分析時，如果將焦點放在非 0 的 area 資料上，從 Fig. 6-2 ~ 6-5 中，可以明顯地看出經過轉換後的資料都呈現出很接近常態的分佈，尤其以 Yeo-Johnson transform 的效果最為顯著。因此，在這筆資料中，我認為使用 Yeo-Johnson transform 是最適合的轉換方式。

# 3. Variables Selection

在變數選擇上，除了引用 P. Cortez *et al. $_{[5]}$* 將資料分為 STFWI, STM, FWI, M 四組外 (Table. 2)，我用了 Wrapper method 中的 Sequential Forward / Backward / Forward floating / Backward floating Selection 共四種方法去找出最重要且適合的變數 $_{[6], [7],[12]}$。

可以看到四條曲線都在約 15 個變數左右時斜率變化趨近平緩，故先挑出各組的影響力前 15 ，其中四組共同擁有的 9 個變數為 [ ”X”, “FFMC”, “DMC”, “ISI”, “temp”, “wind”, “month_dec”, “month_nov”, “day_fri” ]，故將這 9 個變數也列為一組和論文中的其他四組做比較。

| Feature selection | Variables |
| --- | --- |
| STFWI | Spatial + Temporal + FWI |
| STM | Spatial + Temporal + Weather  |
| FWI | FWI |
| M | Weather |

Table. 2

### SFS

![Fig. 7-1](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b28377c2-89e7-4c95-aa3e-3b4eb2561039/newplot.png)

Fig. 7-1

### SBF

![Fig. 7-2](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9588e57e-928c-4ed0-86dc-713384aea74b/newplot.png)

Fig. 7-2

### SFFS

![Fig. 7-3](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/771b22cd-02ce-435d-87c1-1e69b2f5d2ce/newplot.png)

Fig. 7-3

### SBFS

![Fig. 7-4](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8f873eae-0c65-4a05-b84a-c88dab175ceb/newplot.png)

Fig. 7-4

![截圖 2023-04-01 22.52.31.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b9df950d-b138-4a9b-8664-81908763e19b/%E6%88%AA%E5%9C%96_2023-04-01_22.52.31.png)

# 4. Regression

為了要預測火災的燃燒面積，我對資料集進行了監督式學習中的迴歸預測分析，包含 Multiple Linear Regression (MR) 和 Support Vector Regression (SVR)

Multiple Linear Regression 是 Ordinary Least-Square Regression 的延伸，使用多個自變量來預測應變量的結果。 MR 的目標是最小化預測值和實際值之間的平方差來模擬解釋自變量和應變量之間的線性關係 $_{[18]}$，但 SVR 是求係數的 L2-norm 的最小化。Support Vector 的概念簡單來說就是將資料從低維度空間中投影至高維度空間，使原本在低維度無法進行切割的資料，在高維度時能找到超平面來分開樣本 $_{[9], [10]}$，SVR 中的誤差項通過設置誤差上限 $\epsilon$ 來限制，使得預測值和實際值之間的絕對差值小於或等於 $\epsilon$，通過調整 $\epsilon$ 的值，我們可以控制 SVR 模型的準確性。較小的 $\epsilon$ 值意味著模型對誤差的容忍度更嚴格，這會導致 Overfitting，而較大的 $\epsilon$  值會導致 Underfitting。簡單來說，SVR 使我們能夠靈活地定義我們的模型可以接受多少誤差，並找到合適的直線或更高維度的超平面來擬合數據 $_{[19],[20]}$。

$$
L_2\ norm=||x||_2 = (\sum_{i=1}^n |x_i|^2)^{1/2}
$$

## Multiple Linear Regression

### All variables

**Original data**

$$
MAD:0.58
\\
RMSE:0.64
\\
MAE:0.58
\\
R^2:0.04
$$

![Fig. 9-1](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c3cf3d4d-bae7-410a-857f-365b1618ddb5/newplotall.png)

Fig. 9-1

**Upsample data**

$$
MAD:0.38
\\
RMSE:0.47
\\
MAE:0.38
\\
R^2:0.39
$$

![Fig. 9-7](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/24a74111-92df-48fb-add6-b3f76d1d83cb/newplot.png)

Fig. 9-7

**Upsample data without area = 0**

$$
MAD:0.42
\\
RMSE:0.59
\\
MAE:0.42
\\
{\color{red}R^2:0.46}
$$

![Fig. 9-13](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/581fe2c2-4287-4770-a89e-16a91735782f/newplot.png)

Fig. 9-13

### Stepwise selection

**Original data**

$$
MAD:0.55
\\
RMSE:0.60
\\
MAE:0.55
\\
R^2:0.08
$$

![Fig. 9-2](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6eeea79b-7273-4976-8100-c2f54c85852a/newplot.png)

Fig. 9-2

**Upsample data**

$$
MAD:0.41
\\
RMSE:0.49
\\
MAE:0.41
\\
R^2:0.29
$$

![Fig. 9-8](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/647c4785-9c36-4239-b2e5-a5fb5180e6b8/newplot.png)

Fig. 9-8

**Upsample data without area = 0**

$$
MAD:0.57
\\
RMSE:0.72
\\
MAE:0.57
\\
R^2:0.13
$$

![Fig. 9-14](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8ae2de64-8401-431a-89fc-e029c2f2a4d2/newplot.png)

Fig. 9-14

### STFWI

**Original data**

$$
MAD:0.59
\\
RMSE:0.64
\\
MAE:0.59
\\
R^2:0.03
$$

![Fig. 9-3](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c59f78f4-b3dd-4f85-b779-036e04055059/newplotstfwi.png)

Fig. 9-3

**Upsample data**

$$
MAD:0.44
\\
RMSE:0.64
\\
MAE:0.44
\\
R^2:0.40
$$

![Fig. 9-9](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8ca908c7-997a-4f45-8287-0330360bf1ee/newplot.png)

Fig. 9-9

**Upsample data without area = 0**

$$
MAD:0.41
\\
RMSE:0.55
\\
MAE:0.41
\\
R^2:0.43
$$

![Fig. 9-15](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/73a784fc-db7e-495c-9369-107cc4276044/newplot.png)

Fig. 9-15

### STM

**Original data**

$$
MAD:0.59
\\
RMSE:0.65
\\
MAE:0.59
\\
R^2:0.02
$$

![Fig. 9-4](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/94d114cf-7f88-42cf-b716-4dad83d2d83d/newplotstm.png)

Fig. 9-4

**Upsample data**

$$
MAD:0.41
\\
RMSE:0.50
\\
MAE:0.41
\\
R^2:0.36
$$

![Fig. 9-10](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/eccfedee-8c1b-4583-a0fc-020840b4c4e2/newplot.png)

Fig. 9-10

**Upsample data without area = 0**

$$
MAD:0.49
\\
RMSE:0.69
\\
MAE:0.49
\\
R^2:0.35
$$

![Fig. 9-16](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2738b20c-c1f6-4900-8345-4391b1d6c509/newplot.png)

Fig. 9-16

### FWI

**Original data**

$$
MAD:0.62
\\
RMSE:0.67
\\
MAE:0.62
\\
R^2:0.02
$$

![Fig. 9-5](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6fac97a8-bae8-4b2b-93f6-8a381644d23a/newplotfwi.png)

Fig. 9-5

**Upsample data**

$$
MAD:0.54
\\
RMSE:0.58
\\
MAE:0.54
\\
R^2:0.05
$$

![Fig. 9-11](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5ad8aff0-213d-4931-9107-5d36cfa0179a/newplot.png)

Fig. 9-11

**Upsample data without area = 0**

$$
MAD:0.60
\\
RMSE:0.74
\\
MAE:0.60
\\
R^2:0.06
$$

![Fig. 9-17](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5036d30e-18cd-4b83-94bd-71bddca4f257/newplot.png)

Fig. 9-17

### M

**Original data**

$$
MAD:0.59
\\
RMSE:0.63
\\
MAE:0.59
\\
R^2:0.02
$$

![Fig. 9-6](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/00e3987f-20ed-4e5a-a9de-f4529481a613/newplot.png)

Fig. 9-6

**Upsample data**

$$
MAD:0.51
\\
RMSE:0.58
\\
MAE:0.51
\\
R^2:0.14
$$

![Fig. 9-12](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/61fe133f-2330-4516-a01d-386417c13307/newplot.png)

Fig. 9-12

**Upsample data without area = 0**

$$
MAD:0.67
\\
RMSE:0.81
\\
MAE:0.67
\\
R^2:0.02
$$

![Fig. 9-18](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b23ecd8a-3e78-4979-bb9b-8cf8f3ec5878/newplot.png)

Fig. 9-18

---

## Support Vector Regression

### All variables

**Original data**

$$
MAD:0.58
\\
RMSE:0.67
\\
MAE:0.58
\\
R^2:0.01
$$

![Fig. 10-1](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2a45ef03-f185-4005-8b04-95b8c1b1c6f8/newplot.png)

Fig. 10-1

**Upsample data**

$$
MAD:0.25
\\
RMSE:0.38
\\
MAE:0.25
\\
{\color{red}R^2:0.61}
$$

![Fig. 10-7](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/cfeea892-0f04-4030-92b0-214f702e1ba2/newplot.png)

Fig. 10-7

**Upsample data without area = 0**

$$
MAD:0.24
\\
RMSE:0.39
\\
MAE:0.24
\\
{\color{red}R^2:0.73}
$$

![Fig. 10-13](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ec349872-8efe-4227-8493-5ab19237ca67/newplot.png)

Fig. 10-13

### Stepwise selection

**Original data**

$$
MAD:0.60
\\
RMSE:0.71
\\
MAE:0.60
\\
R^2:0.02
$$

![Fig. 10-2](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/82b1495b-475c-464e-bcb0-f348238ef1e8/newplot.png)

Fig. 10-2

**Upsample data**

$$
MAD:0.29
\\
RMSE:0.43
\\
MAE:0.29
\\
R^2:0.52
$$

![Fig. 10-8](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1723a631-51fa-4d23-86f3-32e00f718638/newplot.png)

Fig. 10-8

**Upsample data without area = 0**

$$
MAD:0.31
\\
RMSE:0.51
\\
MAE:0.31
\\
R^2:0.56
$$

![Fig. 10-14](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b413323c-3e0b-4c2b-9b4e-cb621dffd907/newplot.png)

Fig. 10-14

### STFWI

**Original data**

$$
MAD:0.58
\\
RMSE:0.69
\\
MAE:0.58
\\
R^2:0.03
$$

![Fig. 10-3](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/68d2a272-eb32-499e-a16a-d6273ad30527/newplot.png)

Fig. 10-3

**Upsample data**

$$
MAD:0.26
\\
RMSE:0.37
\\
MAE:0.26
\\
{\color{red}R^2:0.63}
$$

![Fig. 10-9](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/583342c3-24cc-488f-9825-85873fcf8484/newplot.png)

Fig. 10-9

**Upsample data without area = 0**

$$
MAD:0.30
\\
RMSE:0.48
\\
MAE:0.30
\\
{\color{red}R^2:0.63}
$$

![Fig. 10-15](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/99d26288-d8e5-4c95-a85e-a10dff47fc26/newplot.png)

Fig. 10-15

### STM

**Original data**

$$
MAD:0.57
\\
RMSE:0.67
\\
MAE:0.57
\\
R^2:0.05
$$

![Fig. 10-4](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f74b433c-c469-4d04-a552-f6145fb45502/newplot.png)

Fig. 10-4

**Upsample data**

$$
MAD:0.25
\\
RMSE:0.38
\\
MAE:0.25
\\
R^2:0.59
$$

![Fig. 10-10](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/24ff5b2d-7ed9-453e-a8ef-938d4757cc0a/newplot.png)

Fig. 10-10

**Upsample data without area = 0**

$$
MAD:0.30
\\
RMSE:0.49
\\
MAE:0.30
\\
{\color{red}R^2:0.65}
$$

![Fig. 10-16](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7b2ab6b2-cefb-4cd2-8e8e-5969ce40858f/newplot.png)

Fig. 10-16

### FWI

**Original data**

$$
MAD:0.63
\\
RMSE:0.76
\\
MAE:0.63
\\
R^2:0.00
$$

![Fig. 10-5](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/606dfe00-3f67-4270-89cf-ac123de3cfd1/newplot.png)

Fig. 10-5

**Upsample data**

$$
MAD:0.39
\\
RMSE:0.54
\\
MAE:0.39
\\
R^2:0.21
$$

![Fig. 10-11](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9d5ae483-7e30-4dca-96ef-3560ad93b526/newplot.png)

Fig. 10-11

**Upsample data without area = 0**

$$
MAD:0.40
\\
RMSE:0.64
\\
MAE:0.40
\\
R^2:0.31
$$

![Fig. 10-17](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bd07da54-67e1-4480-b05e-c33791e08ca8/newplot.png)

Fig. 10-17

### M

**Original data**

$$
MAD:0.57
\\
RMSE:0.69
\\
MAE:0.57
\\
R^2:0.03
$$

![Fig. 10-6](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/606dfe00-3f67-4270-89cf-ac123de3cfd1/newplot.png)

Fig. 10-6

**Upsample data**

$$
MAD:0.35
\\
RMSE:0.51
\\
MAE:0.35
\\
R^2:0.32
$$

![Fig. 10-12](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/71a27202-f78f-4c24-88cd-87cbbbb2fdbf/newplot.png)

Fig. 10-12

**Upsample data without area = 0**

$$
MAD:0.40
\\
RMSE:0.64
\\
MAE:0.40
\\
R^2:0.31
$$

![Fig. 10-18](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/08cd7f2b-f9f6-4367-a5fc-b52d002b42bd/newplot.png)

Fig. 10-18

觀察原始資料集進行的 MR (Fig. 9-1~9-6) 和 SVR (Fig. 10-1 ~ 10-6) 預測表現並不理想。這可能是由於資料過度集中在 0 附近 (50% 的資料 ≤ 0.52)，而另外一半資料的差異範圍卻很大，同時，由於資料集的大小只有 517 筆，在將近一半的資料為 0 且 非 0 資料很少又差異極大的情況下，導致模型難以針對非 0 資料學習並做出正確預測，在一個 Imbalance 的 dataset 中，多數類 (area = 0) 的數據點數量遠遠高於少數類的數量。這可能會導致模型性能的偏差，因為模型傾向於預測多數類比少數類更頻繁。

為了解決這個問題，我參考了處理不平衡資料的常見方法，即使用上採樣 (Data Upsample)，上採樣是透過增加少數類別的樣本數，使得少數類別的樣本數與多數類別相當，進而提高模型預測能力 。我所使用的方法是 Python 中的 random.sample() function $_{[11]}$，在各個月份隨機生成 200 組相近的數據點，使每個月份的數據點數量相近，稱其為 **Upsample by month**。使用這個方法後 (Fig. 9-7 ~ 9-12)，可以看到確實對於每個群體的 $R^2$ 都提升了 3 到 10 倍不等，但是模型仍然受到過多 area = 0 的資料點的影響，導致無法正確預測，可以從結果看到，test set 中 area = 0 的資料點幾乎都不在迴歸線上。

為此，我嘗試使用不包含 area = 0 的資料進行迴歸 (Fig. 9-13 ~ 9-18)，但結果並不如預期。這可能是因為某些月份 (例如 11 月) 的數據結果都是 area = 0，如果排除這些資料點，那麼這些月份的資料就會消失，導致模型產生更大的誤差。

經參考 P. Cortez *et al. $_{[5]}$* 的研究結果，得出 Support Vector Machine (SVM) 可能是最適合這筆資料的 Data mining model。同 MR 分別做了 Upsample (Fig. 10-7 ~ 10-12) 和 Upsample Without area = 0 (Fig. 10-13 ~ 10-18) 兩種處理方式，而確實也相較於 MR 得到的更好的預測結果。最佳的分組是保留所有變數 (Fig. 10-7, 10-13)，有 / 無包含 area = 0 的 $R^2$ 分別達到 0.61 和 0.73，是 MR 在同樣條件下的 1.5 倍，其他分組如 STFWI 和 STM 的 $R^2$ 也達到 0.6 以上，Feature selection 選出的 9 個變數 $R^2$ 也將近 0.6，較原始資料集的預測高上非常多。

# 5. Classification

為了解決無法準確預測 area = 0 的情況，我嘗試將預測從迴歸模型 (Regression) 轉成分類問題 (Classification)，透過將 area 分成 **no fire** (area = 0) / **small fire** (0 < area < 6) / **large fire** (6 < area) 三組，去預測潛在的火勢大小以決定需要調派的資源，可能會比準確預測一個燃燒的面積來得有意義。Upsample by month 前 / 後三組的比例如下 (Fig. 12-1, 12-2)，可以看到這個方法大致以一樣的比例去做 Upsample，但仍會有一點微小的差異。

使用的模型有 Logistic Regression 和 Support Vector Classification。Logistic Regression 根據給定的自變量數據集估計事件發生的概率。 由於結果是機率，因此應變量介於 0 和 1 之間。Logistic Regression 是一個平滑的曲線 (Fig. 11-1) $Z=\beta_0+\beta_1*x_1+......+\beta_n*x_n$，其中會通過 Maximum Likelihood Function (MLE) 多次迭代 (iteration) 試圖最大化該函數以找到最佳參數 $\beta$ ，一旦找到最佳係數，就可以計算每個變量的條件機率，並將它們加在一起以產生預測概論。簡單來說，當 $Z$ 越大時，判斷成 A 類的機率越大，反之判斷成A類的機率越小 $_{[22],[23]}$。

![Fig. 11-1 Logistic Function [21]](http://rasbt.github.io/mlxtend/user_guide/classifier/LogisticRegression_files/logistic_function.png)

Fig. 11-1 Logistic Function [21]

Upsample 的方法先經過前面提到的 Upsample by month 後再透過 Imbalanced learn 中的 RandomOverSampler (ROS) 和 SMOTE 調整，sampling strategy 均令為 “Auto”。

RandomOverSampler 的採樣模式是對少數類別進行隨機抽樣，並進行替換，直到少數類別的樣本數等於多數類別的樣本數。但這個方法也可能導致 Overfitting，如果合成的樣本不能代表少數類別的真實分佈。然而，這是一個快速而簡單的方法來平衡數據集 $_{[7], [13]}$。

SMOTE 的採樣模式是通過在現有的少數類別樣本之間進行插值，為少數類別生成合成樣本。具體來說，它隨機選擇一個少數群體的樣本，並在特徵空間 (feature space) 中找到它的 k 個最近的鄰居 (k nearest neighbors)，然後隨機選擇這些鄰居中的一個，並在連接原始少數群體樣本和所選鄰居的線段上隨機選擇一個點，生成一個新的合成樣本 $_{[14], [15]}$。

量化模型的指標為 Accuracy, Precision, Recall, F1-score, Confusion matrix & ROC curve 

$$
Accuracy=\frac{TP+TN}{ALL} \ \ ;\ \ Precision=\frac{TP}{TP+FP}\ \ ;\ \ Recall=\frac{TP}{TP+FN}\ \ ;\ \ F1=\frac{2*Precision*Recall}{Precision+Recall}
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

ROC curve 的組成 x 軸為 False Positive rate, y 軸為 True Positive rate (Fig. 11-2)，最理想的模型 Area under curve (AUC) = 1 (Green line) 會通過圖的左上角，表示 TPR = 1, FPR = 0，不管閥值 (decision threshold) 為何都可以 100% 預測 ; 當 AUC = 0.5 (Red line) 時，曲線幾乎等於對角線，模型沒有任何鑑別度，因為不管你正樣本分對比例多高，永遠都會有同等比例的負樣本被錯判。簡單來說，當 ROC curve 越接近左上角，表示模型越準確，可以用曲線下面積 AUC 來衡量  $_{[16]}$

![Fig. 11-2 [17]](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3809cae1-c866-4513-a74f-9661c418f18d/assets2F-LvBP1svpACTB1R1x_U42F-LvGspxW3Zko2589SZEN2F-LvHDdtKiSfM4WORukWK2Fimage.webp)

Fig. 11-2 [17]

![Fig. 12-1 (Upsample 前)](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5b4b8f2d-83a4-43dd-b07a-f50c054f6170/%E6%88%AA%E5%9C%96_2023-04-02_17.12.27.png)

Fig. 12-1 (Upsample 前)

![Fig. 12-2 (Upsample 後)](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b649d265-0938-4d5d-ac0f-9a9968b42f79/%E6%88%AA%E5%9C%96_2023-04-02_17.12.57.png)

Fig. 12-2 (Upsample 後)

## Difference between different sampling strategy in imbalanced learn

- Auto: 對所有類別進行重新採樣，使其樣本數與樣本數中位數的類別相同，當希望算法自動平衡類別時，此策略很有用。
- Not minority: 對除少數類別之外的所有類別 (No, Large fire) 進行重新採樣。 該算法將根據需要盡可能刪除多的多數類別樣本。
- Minority: 對少數類別 (Small Fire) 進行重新採樣，使其與多數類別 (No fire) 具有相同數量的樣本。該策略將少數類別的目標比例設置為等於1，意味著算法將根據需要盡可能生成多的少數類樣本以達到該比例。
- Not majority: 對除了多數類別以外的所有類別 (Small, Large fire) 進行重新採樣。意味著 dataset 將是平衡的。 該算法將根據需要盡可能生成多的少數類樣本以實現這種平衡。

**Auto** (Group proportion)

$$
No\ fire:33.33\%
\\
Small\ fire:33.33\%
\\
Large\ fire:33.33\%
\\
F1:0.89
$$

![Fig. 13-1](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bed3c01d-2fcd-4808-97fa-91e087646834/newplot.png)

Fig. 13-1

**Not minority**

$$
No\ fire:44.34\%
\\
Small\ fire:11.32\%
\\
Large\ fire:44.34\%
\\
F1:0.89
$$

![Fig. 13-2](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/da71594e-5448-4c29-8127-cc41c04cbde6/newplot.png)

Fig. 13-2

**Minority**

$$
No\ fire:40.16\%
\\
Small\ fire:40.16\%
\\
Large\ fire:19.68\%
\\
F1:0.89
$$

![Fig. 13-3](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/36b8bb68-3e18-4653-9912-c6f5bbd54647/newplot.png)

Fig. 13-3

**Not majority**

$$
No\ fire:33.33\%
\\
Small\ fire:33.33\%
\\
Large\ fire:33.33\%
\\
F1:0.89
$$

![Fig. 13-4](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/97e670cc-01f1-41f0-8c83-8ac8e6cb9c91/newplot.png)

Fig. 13-4

在 Sampling strategy 的選擇上，不僅僅是希望能補上 Regression 模型難以預測的 No fire 部分，更希望能對 Small fire 和 Large fire 也有很好的預測能力，故在 Strategy 的選擇上選了均衡的 “auto”，而非偏向 Large fire 的 “Not minority” 或偏向 Small fire 的 “Minority”，但這也可以依想預測的類別做調整，並沒有一定的答案。

---

## Logistic Regression

Red curve: No fire ; Green curve: Small fire ; Blue curve: Large fire

### All variables

### Stepwise selection

**RandomOverSampler**

$$
Accuracy:67.20\%
\\
Precision:72.44\%
\\
Recall:67.20\%
\\
F1:0.69
\\
{\color{red}AUC:0.85/0.85/0.84}
$$

![Fig. 14-1](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0e930848-c6ce-4b3e-bb05-fe4356f75be7/newplot.png)

Fig. 14-1

**SMOTE**

$$
Accuracy:66.67\%
\\
Precision:68.90\%
\\
Recall:66.67\%
\\
F1:0.67\\
{\color{red}AUC:0.84/0.83/0.84}
$$

![Fig. 14-7](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/eef62ec1-2bbd-4fe3-b91c-68ca1edf142b/newplot.png)

Fig. 14-7

**RandomOverSampler**

$$
Accuracy:52.65\%
\\
Precision:62.69\%
\\
Recall:52.65\%
\\
F1:0.55\\
AUC:0.75/0.67/0.76
$$

![Fig. 14-2](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d35eaf3c-d48c-4bf6-b5ba-8fb33136b44d/newplot.png)

Fig. 14-2

**SMOTE**

$$
Accuracy:50.53\%
\\
Precision:62.92\%
\\
Recall:50.53\%
\\
F1:0.53\\
AUC:0.74/0.66/0.77
$$

![Fig. 14-8](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1032b925-c619-4eea-a30c-5918f297fb7d/newplot.png)

Fig. 14-8

### STFWI

### STM

**RandomOverSampler**

$$
Accuracy:52.65\%
\\
Precision:62.69\%
\\
Recall:52.65\%
\\
F1:0.55\\
AUC:0.75/0.67/0.76
$$

![Fig. 14-3](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d35eaf3c-d48c-4bf6-b5ba-8fb33136b44d/newplot.png)

Fig. 14-3

**SMOTE**

$$
Accuracy:50.53\%
\\
Precision:62.92\%
\\
Recall:50.53\%
\\
F1:0.53\\
AUC:0.74/0.66/0.77
$$

![Fig. 14-9](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1032b925-c619-4eea-a30c-5918f297fb7d/newplot.png)

Fig. 14-9

**RandomOverSampler**

$$
Accuracy:64.68\%
\\
Precision:71.82\%
\\
Recall:64.68\%
\\
F1:0.67\\
{\color{red}AUC:0.85/0.82/0.81}
$$

![Fig. 14-4](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b50f73bf-e69e-457c-95e6-349c88f29fa0/newplot.png)

Fig. 14-4

**SMOTE**

$$
Accuracy:64.55\%
\\
Precision:68.27\%
\\
Recall:64.55\%
\\
F1:0.66\\
{\color{red}AUC:0.83/0.81/0.81}
$$

![Fig. 14-10](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6acc9689-257a-4d07-9ff7-49c46aeec128/newplot.png)

Fig. 14-10

### FWI

### M

**RandomOverSampler**

$$
Accuracy:54.89\%
\\
Precision:63.52\%
\\
Recall:54.89\%
\\
F1:0.57\\
AUC:0.66/0.69/0.67
$$

![Fig. 14-5](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fadb73ed-561b-4b17-8e41-0af6bc0272d5/newplot.png)

Fig. 14-5

**SMOTE**

$$
Accuracy:56.08\%
\\
Precision:66.28\%
\\
Recall:56.08\%
\\
F1:0.58\\
AUC:0.66/0.68/0.67
$$

![Fig. 14-11](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/83e82d12-549a-4e35-9313-e87b36069e9a/newplot.png)

Fig. 14-11

**RandomOverSampler**

$$
Accuracy:44.31\%
\\
Precision:50.24\%
\\
Recall:44.31\%
\\
F1:0.46\\
AUC:0.65/0.60/0.65
$$

![Fig. 14-6](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f2aa577f-d9bb-464c-bef5-fb52d415c2b1/newplot.png)

Fig. 14-6

**SMOTE**

$$
Accuracy:40.21\%
\\
Precision:48.17\%
\\
Recall:40.21\%
\\
F1:0.42\\
AUC:0.66/0.62/0.65
$$

![Fig. 14-12](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/315b4791-4fce-4b59-af57-922564d2fe9f/newplot.png)

Fig. 14-12

---

## Support Vector Classification

### All Variables

### Stepwise selection

**RandomOverSampler**

$$
Accuracy:89.02\%
\\
({\color{red}96.70\%}/74.11\%/81.82\%)
\\
Precision:89.15\%
\\
Recall:89.02\%
\\
F1:0.89
$$

![Fig. 15-1](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0c42cd07-73bc-48aa-8915-3d35091ac55e/newplot.png)

Fig. 15-1

**SMOTE**

$$
Accuracy:89.42\%
\\
({\color{red}95.75\%}/77.68\%/81.82\%)
\\
Precision:89.54\%
\\
Recall:89.42\%
\\
F1:0.89
$$

![Fig. 15-7](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5e316a77-1c80-4fdd-b963-d40922da8c98/newplot.png)

Fig. 15-7

**RandomOverSampler**

$$
Accuracy:90.34\%
\\
({\color{red}97.25\%}/76.56\%/82.46\%)
\\
Precision:90.44\%
\\
Recall:90.34\%
\\
{\color{red}F1:0.91}
$$

![Fig. 15-2](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a83ec6c3-8b64-4c32-b131-1f9dec95bf01/newplot.png)

Fig. 15-2

**SMOTE**

$$
Accuracy:89.68\%
\\
({\color{red}96.25\%}/78.91\%/83.33\%)
\\
Precision:89.61\%
\\
Recall:89.68\%
\\
{\color{red}F1:0.90}
$$

![Fig. 15-8](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b5cfbf83-d5a9-48ce-a97f-ee1751c45d26/newplot.png)

Fig. 15-8

### STFWI

### STM

**RandomOverSampler**

$$
Accuracy:87.70\%
\\
({\color{red}91.63\%}/77.27\%/{\color{red}86.70\%})
\\
Precision:87.94\%
\\
Recall:87.70\%
\\
F1:0.88
$$

![Fig. 15-3](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ead88f13-aaf4-4414-a4df-4824468cd685/newplot.png)

Fig. 15-3

**SMOTE**

$$
Accuracy:86.90\%
\\
(89.90\%/81.82\%/{\color{red}85.32\%})
\\
Precision:87.48\%
\\
Recall:86.90\%
\\
F1:0.87
$$

![Fig. 15-9](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/634cbe26-4a33-47f2-a614-d456a934b19b/newplot.png)

Fig. 15-9

**RandomOverSampler**

$$
Accuracy:88.23\%
\\
({\color{red}91.79\%}/84.88\%/83.83\%)
\\
Precision:88.93\%
\\
Recall:87.83\%
\\
F1:0.88
$$

![Fig. 15-4](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1c4709e4-6ec1-407e-8e3b-47f902ca953b/newplot.png)

Fig. 15-4

**SMOTE**

$$
Accuracy:88.36\%
\\
({\color{red}94.00\%}/77.48\%/83.33\%)
\\
Precision:88.48\%
\\
Recall:88.36\%
\\
F1:0.88
$$

![Fig. 15-10](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4a4b925b-7657-46f3-a579-0fb057084a89/newplot.png)

Fig. 15-10

### FWI

### M

**RandomOverSampler**

$$
Accuracy:82.54\%
\\
(84.06\%/76.19\%/83.33\%)
\\
Precision:83.06\%
\\
Recall:82.54\%
\\
F1:0.83
$$

![Fig. 15-5](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/40af6d6d-bb76-4485-8fe5-20b0bfdec37e/newplot.png)

Fig. 15-5

**SMOTE**

$$
Accuracy:82.67\%
\\
(82.84\%/80.95\%/80.18\%)
\\
Precision:83.92\%
\\
Recall:82.67\%
\\
F1:0.83
$$

![Fig. 15-11](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/022983f5-92ee-4738-921c-bcb947a92d69/newplot.png)

Fig. 15-11

**RandomOverSampler**

$$
Accuracy:89.02\%
\\
({\color{red}93.65\%}/79.20\%/{\color{red}86.50\%})
\\
Precision:88.97\%
\\
Recall:89.02\%
\\
F1:0.89
$$

![Fig. 15-6](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d1514964-58ff-445c-86ab-d61d86770dcc/newplot.png)

Fig. 15-6

**SMOTE**

$$
Accuracy:88.49\%
\\
({\color{red}93.65\%}/77.60\%/{\color{red}85.65\%})
\\
Precision:88.46\%
\\
Recall:88.49\%
\\
F1:0.88
$$

![Fig. 15-12](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6d47ac18-693d-4a59-999f-e46bcf4aad07/newplot.png)

Fig. 15-12

根據 Classification 的分析結果 (Fig. 14, 15)，可以看出 **SVC** 的表現明顯優於 Logistic Regression。Logistic Regression 中最好的模型是**保留所有變數** (Fig. 14-1, 14-7)，AUC 為 **0.83** 或更高，但對於 FWI 和 M 兩組而言，AUC 僅只有 0.6 左右，幾乎等於隨機猜測。此外， 針對綜合評估 Precision 和 Recall 的 F1 score，Logistic Regression 普遍不及 SVC 模型。值得注意的是，SVC 模型的 F1 score 除了 FWI 那組外均達到 **0.88 以上**，且幾乎對 No fire 的識別率都達到 **90%** 以上。在 SVC 的所有模型中，Stepwise selection 的 9 個變量表現最佳，F1 score 為 **0.91**。該模型可以成功識別 **97.25%** 的 No fire，對 Small fire 和 Large fire 分別也有將近 8 成和 8 成以上的準確率。它不僅成功彌補了 Regression 模型難以預測的 area = 0 的缺陷，同時也展現出對其他類別的出色識別能力。

# 6. Conclusion

回歸到問題本身，Classification 中的 SVC 模型可以準確識別 No fire 也就是 area = 0 的部分。若要預測火災發生面積，可以採用**保留所有變數且不包含 area = 0 的  SVR 模型**，儘管這部分仍然需要進一步優化 ; 對於僅需要大致預測火災規模的需求，**SVC 中的 Stepwise selection 模型**已經可以達到 90% 以上的精準度，而其他模型也達到了 88% 以上的水準。若需要精準的預測 Small fire, Large fire 的火災發生面積，則可以採用 Classification 後再在各類別內做 Regression，這是一個可以進一步延伸的方向。

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
