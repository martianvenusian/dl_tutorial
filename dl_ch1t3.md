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

Payqagan bo'lsangiz *requires_grad+True* aggementini ishlatdik. Bu argument PyTorchga Amallarni bajarish davomida hosil bo'lgan tensorning butun shajara tarixini yodda saqlab borishe kerakligini aytadi. 