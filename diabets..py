import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import  StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  classification_report
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.cluster import KMeans

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df=pd.read_csv("datasets/diabetes.csv")

#VERİ ANALİZİ
print(df.info())
print(df.shape)
# Veri setindeki değişken ve gözlem sayısı
print("örnek sayısı : " ,len(df))
print("özellik sayısı : ", len(df.columns))


#VERİ GÖRSELLEŞTİRME
#target bağımlı değişkenin sınıflarının dağılımı ve görseli
print(df["Outcome"].value_counts())
sns.countplot(x="Outcome", data=df)
plt.show()

#bağımsız değişkenlerin gösterimi (histogram ile)
cols = [col for col in df.columns if "Outcome" not in col]

def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    #plt.show(block=True)
for col in cols:
    plot_numerical_col(df, col)

'''def plot_numerical_col_boxplot(dataframe, numerical_col):
    plt.figure(figsize=(8, 6)) # boxplot boyutunu ayarla
    dataframe.boxplot(column=numerical_col)
    plt.ylabel(numerical_col)
    plt.title("Boxplot of " + numerical_col)
    plt.show(block=True) # diğer görselleri engelle

for col in cols:
    plot_numerical_col_boxplot(df, col)'''
#VERİ ÖN İŞLEME

#eksik değer tespiti:
print(df.isnull().sum())
print(df.describe().T)
#aykırı değer tespiti
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
for col in cols:
    print(col,check_outlier(df,col))
#aykırı değerleri belirlenen eşik değerleri ile değiştirme
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df, "Insulin")



#KNN ALGORİTMASI

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
#standart sclaer ile standartlaştırma yapıldı
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


knn_model = KNeighborsClassifier().fit(X, y)
# Confusion matrix için y_pred:
y_pred = knn_model.predict(X)

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)

#modeli kurduğumuz veride test ettik:
print(classification_report(y, y_pred))
#model eğitim-test rassal %66-34 olacak şekilde
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.34, random_state=17)
knn_model= KNeighborsClassifier().fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
plot_confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

#cv yöntemi 5 katlı
cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

print(f"Ortalama Test Accuracy: {cv_results['test_accuracy'].mean()*100:.2f}%")
print(f"Ortalama Test F1 Score: {cv_results['test_f1'].mean()*100:.2f}%")
print(f"Ortalama Test ROC AUC Score: {cv_results['test_roc_auc'].mean()*100:.2f}%")


#K-MEANS ALGORİTMASI
# K-Means uygulama
kmeans = KMeans(n_clusters=2, random_state=42)  # İki küme oluşturduk çünkü "Outcome" sütunu iki sınıfa sahiptir
kmeans.fit(X_scaled)

# Küme merkezlerini ve küme etiketlerini alıyoruz
cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Küme merkezlerini ve küme etiketlerini görselleştirme
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', color='red', s=200)
plt.title('K-Means Kümeleme')
plt.xlabel('Scaled Feature 1')
plt.ylabel('Scaled Feature 2')
plt.show()

