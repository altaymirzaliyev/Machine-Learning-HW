import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


data=pd.read_csv("Dataset/hw1Data.txt", header=None)

#%% Veri Görselleştirme
s1 = data.iloc[:, 0]  # 1. sınav notu (x ekseni)
s2 = data.iloc[:, 1]  # 2. sınav notu (y ekseni)
labels = data.iloc[:, 2]  # Son sütun (işe kabul: 1, red: 0)


# Grafik için farklı renkler belirleyin
# Kabul edilenler (1) için mavi, red edilenler (0) için kırmızı renkler kullanalım
plt.figure(figsize=(8, 6))

# Kabul edilenler (1) için mavi noktalar
plt.scatter(s1[labels == 1], s2[labels == 1], color='blue', label='Kabul Edilenler')

# Red edilenler (0) için kırmızı noktalar
plt.scatter(s1[labels == 0], s2[labels == 0], color='red', label='Red Edilenler')

# Başlık ve etiketler
plt.title('Sınav Notlarına Göre İşe Alım Durumu')
plt.xlabel('1. Sınav Notu')
plt.ylabel('2. Sınav Notu')

plt.legend()
plt.show()


plt.figure(figsize=(8, 6))
#%% Veri Setini Eğitim, Doğrulama ve Test aşamalarına bölme
X = data.iloc[:, :-1]  # Giriş sütunları
y = data.iloc[:, -1]   # Son sütun (etiket)

# Veriyi eğitim (%60), doğrulama (%20), test (%20) olarak bölelim
# İlk olarak, eğitim ve doğrulama kümelerini ayırıyoruz (%80)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42)

# Sonra, doğrulama ve test kümelerini ayırıyoruz (%50'si doğrulama, %50'si test, yani toplamda %20 test ve %20 doğrulama)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

#Veri Normalizasyonu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)


# Sonuçları kontrol edelim
print(f"Eğitim verisi boyutu: {X_train.shape[0]} örnek")
print(f"Doğrulama verisi boyutu: {X_val.shape[0]} örnek")
print(f"Test verisi boyutu: {X_test.shape[0]} örnek")

#%%# Sigmoid Fonksiyonunun Tanımlanması
def sigmoid(z):
    # Sigmoid fonksiyonu tanımlanıyor
    y_pred = 1/(1+np.exp(-z))
    return y_pred

#%%Cross Entropy Loss function Tanımlanması

def compute_loss(y_train, y_pred):
    # Kaybı hesaplamak için fonksiyon
    m = len(y_train)  # Eğitim örneklerinin sayısı
    loss = -(1/m) * np.sum(y_train * np.log(y_pred) + (1 - y_train) * np.log(1 - y_pred))
    return loss

#%% Stochastic Gradient Descent Fonksiyonun Tanımlanması
learning_rate = 0.01
epochs = 200
# Epoch başına eğitim ve doğrulama kayıplarını hesaplıyoruz
def stochastic_gradient_descent(X_train, y_train, X_val, y_val, learning_rate, epochs):
    m, n = X_train.shape  # m: örnek sayısı, n: özellik sayısı
    w = np.full((n,1),0.01)  # Ağırlıkları başlatıyoruz
    losses_train = []  # Eğitim kayıplarını takip etmek için liste
    losses_val = []    # Doğrulama kayıplarını takip etmek için liste

    for epoch in range(epochs):
        # Eğitim döngüsü
        for i in range(m):
            xi = np.array(X_train[i:i+1]).reshape(1, -1)  # Tek bir örnek
            yi = np.array(y_train[i:i+1]).reshape(1, -1)  # Gerçek etiket
            y_pred = sigmoid(np.dot(xi, w))  # Sigmoid ile tahmin yapıyoruz
            loss = compute_loss(yi, y_pred)  # Kaybı hesaplıyoruz
            gradient = (np.dot(xi.T, (y_pred - yi))) / xi.shape[1]  # Gradien'i hesaplıyoruz
            w -= learning_rate * gradient  # Ağırlıkları güncelliyoruz

        # Eğitim kaybını kaydediyoruz
        losses_train.append(loss)
        
        # Doğrulama kaybını hesaplıyoruz
        y_val_pred = sigmoid(np.dot(X_val, w))  # Doğrulama seti için tahmin yapıyoruz
        if len(y_val_pred.shape) > 1:
            y_val_pred = y_val_pred.ravel() # 1D boyutuna dönüştürüyoruz
        val_loss = compute_loss(y_val, y_val_pred)  # Doğrulama kaybını hesaplıyoruz
        losses_val.append(val_loss)
        
        print(f'Epoch {epoch + 1}/{epochs}, Eğitim Kaybı: {loss}, Doğrulama Kaybı: {val_loss}')
    
    return w, losses_train, losses_val

# Modeli eğitiyoruz ve kayıpları çiziyoruz
w, losses_train, losses_val = stochastic_gradient_descent(X_train, y_train, X_val, y_val, learning_rate, epochs)

#Cost Fonksiyonu Çizdirilmesi
plt.plot(losses_train, color='blue')
plt.title(' Train Datası için Loss Fonksiyonu')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.legend()
plt.show()

# Eğitim ve doğrulama kayıplarını çiziyoruz
plt.figure(figsize=(10, 6))
plt.plot(losses_train, label='Eğitim Kaybı', color='blue')
plt.plot(losses_val, label='Doğrulama Kaybı', color='red')
plt.title(' Eğitim ve Doğrulama Kaybı Karşılaştırılması')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.legend()
plt.show()

#%% Tahmin Fonksiyonunun Tanımlanması
def predict(X, w):
    # X: Giriş verisi (özellikler)
    # w: Öğrenilen ağırlıklar
    
    # Sigmoid fonksiyonunu kullanarak tahmin yapıyoruz
    y_pred = sigmoid(np.dot(X, w))
    
    # Çıktıyı 0 veya 1 olarak sınıflandırıyoruz (ikili sınıflama)
    y_pred_class = (y_pred >= 0.5).astype(int)
    
    return y_pred_class.ravel()

# Eğitim, doğrulama ve test verileri için tahmin yapalım
y_train_pred = predict(X_train, w)
y_val_pred = predict(X_val, w)
y_test_pred = predict(X_test, w)


print("Test tahminleri: ", y_test_pred[:20])

#%% # Accuracy, Precision, Recall ve F1-Score hesaplama fonksiyonları
def compute_accuracy(y_true, y_pred):
    # Doğruluk oranını hesaplamak için fonksiyon
    y_true = y_true.ravel()  # 1D'ye çeviriyoruz
    y_pred = y_pred.ravel()  # 1D'ye çeviriyoruz
    correct = np.sum(y_true == y_pred)  # Doğru tahmin edilen örnek sayısı
    total = len(y_true)  # Toplam örnek sayısı
    return correct / total

def compute_precision(y_true, y_pred):
    # Precision (kesinlik) hesaplamak için fonksiyon
    y_true = y_true.ravel()  # 1D'ye çeviriyoruz
    y_pred = y_pred.ravel()  # 1D'ye çeviriyoruz
    true_positive = np.sum((y_true == 1) & (y_pred == 1))  # TP
    false_positive = np.sum((y_true == 0) & (y_pred == 1))  # FP
    if true_positive + false_positive == 0:
        return 0  # Bölme hatasından kaçınmak için
    return true_positive / (true_positive + false_positive)

def compute_recall(y_true, y_pred):
    # Recall (duyarlılık) hesaplamak için fonksiyon
    y_true = y_true.ravel()  # 1D'ye çeviriyoruz
    y_pred = y_pred.ravel()  # 1D'ye çeviriyoruz
    true_positive = np.sum((y_true == 1) & (y_pred == 1))  # TP
    false_negative = np.sum((y_true == 1) & (y_pred == 0))  # FN
    if true_positive + false_negative == 0:
        return 0  # Bölme hatasından kaçınmak için
    return true_positive / (true_positive + false_negative)

def compute_f1_score(precision, recall):
    # F1-Score hesaplamak için fonksiyon
    if precision + recall == 0:
        return 0  # Bölme hatasından kaçınmak için
    return 2 * (precision * recall) / (precision + recall)


# Eğitim, doğrulama ve test verileri için değerlendirme metriklerini hesaplıyoruz
accuracy_train = compute_accuracy(y_train, y_train_pred)
accuracy_val = compute_accuracy(y_val, y_val_pred)
accuracy_test = compute_accuracy(y_test, y_test_pred)

precision_train = compute_precision(y_train, y_train_pred)
precision_val = compute_precision(y_val, y_val_pred)
precision_test = compute_precision(y_test, y_test_pred)

recall_train = compute_recall(y_train, y_train_pred)
recall_val = compute_recall(y_val, y_val_pred)
recall_test = compute_recall(y_test, y_test_pred)

f1_train = compute_f1_score(precision_train, recall_train)
f1_val = compute_f1_score(precision_val, recall_val)
f1_test = compute_f1_score(precision_test, recall_test)

# Sonuçları yazdıralım
print(f"Eğitim seti - Doğruluk: {accuracy_train}, Kesinlik: {precision_train}, Duyarlılık: {recall_train}, F1-Score: {f1_train}")
print(f"Doğrulama seti - Doğruluk: {accuracy_val}, Kesinlik: {precision_val}, Duyarlılık: {recall_val}, F1-Score: {f1_val}")
print(f"Test seti - Doğruluk: {accuracy_test}, Kesinlik: {precision_test}, Duyarlılık: {recall_test}, F1-Score: {f1_test}")

import matplotlib.pyplot as plt

# Sonuç verilerini .5f formatında hazırlıyoruz
columns = ["Küme", "Doğruluk", "Kesinlik (Precision)", "Duyarlılık (Recall)", "F1-Score"]
data = [
    ["Eğitim", f"{accuracy_train:.5f}", f"{precision_train:.5f}", f"{recall_train:.5f}", f"{f1_train:.5f}"],
    ["Doğrulama", f"{accuracy_val:.5f}", f"{precision_val:.5f}", f"{recall_val:.5f}", f"{f1_val:.5f}"],
    ["Test", f"{accuracy_test:.5f}", f"{precision_test:.5f}", f"{recall_test:.5f}", f"{f1_test:.5f}"]]

# Matplotlib tablosu
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=data, colLabels=columns, cellLoc='center', loc='center', colLoc='center')

# Stil ayarları
table.auto_set_font_size(False)
table.set_fontsize(12)
table.auto_set_column_width(col=list(range(len(columns))))
table.scale(1, 1.5)  # Daha geniş hücreler için ölçeklendirme

# Başlık ve stil
plt.title("Model Performans Sonuçları", fontsize=16, weight='bold', color="darkblue")
plt.figtext(0.5, 0.01, "Tüm sonuçlar 5 ondalık basamakta gösterilmiştir.", wrap=True, 
            horizontalalignment='center', fontsize=10, color="gray")
plt.show()


# Sonuçları bir sözlük olarak tanımlıyoruz
results = {
    "Küme": ["Eğitim", "Doğrulama", "Test"],
    "Doğruluk": [accuracy_train, accuracy_val, accuracy_test],
    "Kesinlik (Precision)": [precision_train, precision_val, precision_test],
    "Duyarlılık (Recall)": [recall_train, recall_val, recall_test],
    "F1-Score": [f1_train, f1_val, f1_test]
}

# Pandas DataFrame oluşturuyoruz
results_df = pd.DataFrame(results)

# Tabloyu yazdırıyoruz
print(results_df)

#%% Confusion Matrix 

# Confusion matrix hesaplanması
confusion_train = confusion_matrix(y_train, y_train_pred)
confusion_val = confusion_matrix(y_val, y_val_pred)
confusion_test = confusion_matrix(y_test, y_test_pred)

# Confusion matrisin görselleştirilmesi
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Eğitim Seti için Confusion Matris
ConfusionMatrixDisplay(confusion_matrix=confusion_train, display_labels=["Red", "Kabul"]).plot(ax=axes[0], cmap='Blues')
axes[0].set_title("Eğitim Seti")

# Validasyon Seti için Confusion Matris
ConfusionMatrixDisplay(confusion_matrix=confusion_val, display_labels=["Red", "Kabul"]).plot(ax=axes[1], cmap='Oranges')
axes[1].set_title("Doğrulama Seti")

# Test Seti için Confusion Matris
ConfusionMatrixDisplay(confusion_matrix=confusion_test, display_labels=["Red", "Kabul"]).plot(ax=axes[2], cmap='Greens')
axes[2].set_title("Test Seti")

plt.tight_layout()
plt.show()

# %%
