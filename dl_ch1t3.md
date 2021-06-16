# Gradientni avtamatik hisoblash

<br/>

### Bu bo'lim nimani o'rgatadi
 *  

### Hal qilinishi kerak bo'lgan muammo
 *  

<br/>

* * *

<br/>

O'tgan darslarimizda kichgana misal yordamida *teskari targ'ibot*ga (*backpropagation*ga) misol ko'rgan edik: biz funksiyalarning tegishli parameterlariga nisbatan uning tarkibiy qismlarinng gradientlarini hisoblagandik (*mode* va *loss*). Bu ishni *zanjir qoidasi*(*chain rule*)dan foydalanib fuksiyaning hosilalarini teskari targ'ib qilish orqali amalga oshirdik. Oldinroq biz "*loss*ning o'zgarish darajasi" deb nomlagan gradientni mos parameterlariga nisbatan bir urunishda hisoblagan edik.

Agar bizda millionlab parameterlarga ega murakkab model bo'lsa ham, bu model farqlanuvchan bo'lar ekan, uning mos parameterlari nisbatan *loss*ining gradientini hisoblash hosilalar uchun analitik ifodalarni yozish va ularni bir marta baholashga anglatadi. Chiziqli va chiqiqli bo'lmagan funksiyalar uchun juda chuqur tarkibiy hosilalarini analitik ifodasini yozish juda maroqli ish emas. Qolarversa anchagina vaqtni talab qiladi.


### **1-QADAM:** Gradientni avtomatik hisoblash

Gradientni avtomatik hisoblashda bizga PyTorch tensorining *autograd* deb ataluvchi tarkibiy qismi (*companent*i) qo'l keladi. PyTorchning tensorlari o'zining qayerdan kelib chiqqanini eslab qoladi. Shu jumladan bajarilgan amallar va o'zi kelib chiqqan ajdod tensorlar, va ular kiritilgan ma'lumotlariga nisbatan bajarilgan amallarning hosilasining zanjirini avtomatik tarzda taqdim qila oladi. Bu degani biz modelni qo'lda hosilasini hisoblashimizga hojot qolmaydi. Qanchalik murakkab bo'lmasin PyTorch berilgan ifodalarning gradientini mos parameterlariga nisbatan avtomatik hisoblay oladi

#### AUTOGRADni qo'llash.

Bir eslab ko'raylik. Bizning *model* va *loss*larimiz quydagicha edi:

```python
# In [4]
def model(x, w, b):
    return w * x + b
```

```python
# In [5]
def loss_fn(yy, y):
    squared_diffs = (yy - y)**2
    return squared_diffs.mean()
```    

Keling parameterlarimizni qayta e'lon qilib olamiz:

```python
# In [6]
params = torch.tensor([1.0, 0.0], requires_grad=True)
```

#### GRAD atributidan foydalanish

Payqagan bo'lsangiz *requires_grad+True* aggementini ishlatdik. Bu argument PyTorchga Amallarni bajarish davomida hosil bo'lgan tensorning butun shajara tarixini yodda saqlab borishi kerakligini aytadi. Boshqa so'z bilan aytganda, biz *params* deb nomlagan tensorning avlodi bo'lgan har qanday tensor o'z ajdodi *params* tensordan shu tensorning o'zigacha bo'lgan barcha funksiyalar ketma-ketligini qayta bog'lanishga huqiqli bo'ladi. Agar o'sha funksiyalar farqlanuvchan bo'lsa (deyarli PyTorch tensor amallari shunday bo'ladi), hosilaning qiymati avtomitik tarzda *param* tensorining atributi *grad* sifatida to'ldiriladi.

Umuman olganda, barcha PyTorch tensorlari *grad* atributiga ega. Odatda u *None* bo'ladi:

```python
# In [7]
params.grad is None
# Out [7] 
# True
```
Bizni yagona qiladigan ishimiz bu biror tensorning *requires_grad*ni *True* qilib belgilash, keyin modelni chaqirib *loss* hisoblash va keyin *loss* tensorning *backward* funksiyani chaqirishdan iboratdir:

```python
# In [8]
loss = loss_fn(model(x, *params), y)
loss.backward()
params.grad
 # Out [8]
 # tensor([6852.5474,   90.4160])
 ```

 Shu o'rinda, *params*ning *grad* atributi *loss*ning parameterlariga nisbatan hosilalarini o'zida saqlaydi. Biz qachonki *w* va *b* parameterlar gradientni talab qilgan holda *loss* ni hisoblaganimizda PyTorch amalllar ketma-ketligga ega bo'lgan gradientni avtomatik hisoblaydigan  zanjirli chizma yaratadi. Biz qachonki *loss.backward()* ni chaqirganimizda, PyTorch shu zanjirli chizma bo'ylab teskari yo'nalishda harakatlanib gradientlarni hasoblaydi.

#### GRAD funksiyalarini to'plash

Biz istagan raqamga raqamli va *requires_grad* True bo'lgan tensorga yoki har qanday funksiyalar to'plamiga ega bo'lishimiz mumkin. Shunday holatda, PyTorch *loss*ning hosilasini funksiyalar zanjiri bo'ylab hisoblab chiqadi va ularning qiymatlarini tensorlar(zanjirli tizimning tugunlari)ning *grad* atributida to'playdi.

##### DIQQAT!
> *backward*ni chaqirish hosilalarni zanjir tugunlarida to'plashga undaydi. Parameterlar yangilanishi uchun *backward*ni ishlatgandan keyin gradientni nolga aylantirishimiz kerak.

Keling bularni yana qaytaraylik: *backward*ni chaqirish hosilalarni zanjir tugunlarida to'planishiga olib keladi. Agar *backward* ertaroq chiqirilgan bo'lsa va yana *backward*ni chaqirsak va har bir tugundagi gradient bundan oldin hisoblanganlarining ustuga qayta to'planadi va bu gradientning qiymati xato bo'lishiga olib keladi.

Buning oldini olish maqsadida biz har bir takrorlanishda parameterlarning gradientini nolga aylantirishimiz kerak. Buni *zero_* metodini ishlatgan holda osongina amalga oshiramiz:

```python
# In [9]
if params.grad is not None:
    params.grad.zero_()
```

##### ESLATMA
> Siz balki nega gradientni nolga teglash *backward* har safar chaqirilganda avtomatik tarzda amalga oshirilmayapti deb qiziqayotgan bo'lishingiz mumkin. Bu ishni bu tarzda qilish ko'p qulaylik yaratadi va murakkab modellarning gradientlari bilan ishlaganda tazorat qilishga imkon beradi.

Ushbu eslatmani meyamizga quyib olgan holda, bizning gradientni avtomatik hisoblaydigan o'rgatish kodimiz qanaqa bo'lishini ko'rib chiqamiz:

```python
# In [10]
def training_loop(n_epochs, learning_rate, params, x, y):
    for epoch in range(1, n_epochs + 1):
        if params.grad is not None:
            params.grad.zero_()
        yy = model(x, *params)
        loss = loss_fn(yy, y)
        loss.backward()
        
        with torch.no_grad():
            params -= learning_rate * params.grad
        
        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
        
    return params
```

Lekin endi bu ish beradimi yo'qmi ko'ramiz:

```python
# In [11]
params = training_loop(
    n_epochs = 5000,
    learning_rate = 1e-2,
    params = torch.tensor([1.0, 0.0], requires_grad=True),
    x = x * 0.1,
    y = y)

# Epoch 500, Loss 0.263147
# Epoch 1000, Loss 0.262983
# Epoch 1500, Loss 0.262982
# Epoch 2000, Loss 0.262982
# Epoch 2500, Loss 0.262982
# Epoch 3000, Loss 0.262982
# Epoch 3500, Loss 0.262982
# Epoch 4000, Loss 0.262982
# Epoch 4500, Loss 0.262982
# Epoch 5000, Loss 0.262982
```

Va nihoyat ohirgi yangilangan parameterdan foydalanib model yordamida *yy*ning ohirgi natijasini ko'ramiz.

```python
# In [11]
x = x * 0.1
yy = model(x, *params)
yy

# Out [12]
# tensor([3.5057, 6.8616, 7.4111, 0.8906, 5.3498, 2.8690, 8.6564, 9.5340, 5.7443, 0.6913, 1.3535], grad_fn=<AddBackward0>)
```

Natija esa bundan oldingi darsligimizda olinga natija bilan bir xil. Bu degani endi ortiq hosilalni qo'lda hisoblashga hojat qolmadi. Juda soz!

</br>

### **2-QADAM:** Natijani tasvirda ifodalash

Natijani tasvir yordamida ham ko'rishimiz mumkin.

```python
# In [13]
%matplotlib inline
from matplotlib import pyplot as plt
fig = plt.figure(dpi=600)
plt.xlabel("x label")
plt.ylabel("y label")
plt.plot(x.numpy(), yy.detach().numpy())
plt.plot(x.numpy(), y.numpy(), 'o')
plt.plot(x.numpy(), yy.detach().numpy(),'o')

plt.savefig("dl_ch1t3_plot01.png", format="png")
```

![data_plot](https://martianvenusian.github.io/dl_tutorial/codes/tutorial_1/dl_ch1t3_plot01.png)
