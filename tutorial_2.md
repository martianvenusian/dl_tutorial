# Me'yorlashtirish (*Normalizing*)
## Bu bo'lim nimalarni o'rgatadi
 * Ma'lumot (data)lari me'yorlash (*normalizing*)

## Hal qilinishi kerak bo'lgan muammo
 * Me'yorlash uslubi yordamida modelni optimalligini oshirish 

<br/>

* * *

<br/>

## **1-QADAM:** Ma'lumotlarni me'yorlash

Bizda ma'lum bir o'rgatishni amalga oshirdek deylik (o'tgan darslikka qarang). Bizning o'rganish jarayonimizda *loss*ning qiymadi juda sekin kamaya boshladi va oqibatda kamayishdan to'xtadi.
```python
# Epoch 1, Loss 3111.259521
# 	 Params:  tensor([ 0.3147, -0.0090])
# 	 Grad:  tensor([6852.5474,   90.4160])
# Epoch 2, Loss 187.329056
# 	 Params:  tensor([ 0.1468, -0.0112])
# 	 Grad:  tensor([1679.8218,   21.8531])
# ...
# Epoch 96, Loss 0.387544
# 	 Params:  tensor([ 0.0922, -0.0081])
# 	 Grad:  tensor([ 0.0054, -0.4098])
# Epoch 97, Loss 0.387527
# 	 Params:  tensor([ 0.0922, -0.0080])
# 	 Grad:  tensor([ 0.0054, -0.4098])
# Epoch 98, Loss 0.387510
# 	 Params:  tensor([ 0.0922, -0.0080])
# 	 Grad:  tensor([ 0.0054, -0.4097])
# Epoch 99, Loss 0.387493
# 	 Params:  tensor([ 0.0922, -0.0080])
# 	 Grad:  tensor([ 0.0054, -0.4097])
# Epoch 100, Loss 0.387477
# 	 Params:  tensor([ 0.0922, -0.0079])
# 	 Grad:  tensor([ 0.0054, -0.4097])
```
Bunday hollarda modelning optimalligini oshirish va *loss*ning qiymatini yanada minimallashtirishda ishlatiladigan uslublardan biri bu ***me'yorlashtirish** (***`Normalizing`***)dir

Yuqoridagi natijadaning birinchi *epoch*iga qaraydigan bo'lsak *weight*ning gradienti *bias*ning gradientidan 75 marta katta ekanini ko'ramiz. Bu shuni bildiradiki *weight* bilan *bies* turli o'lchovlarda. Birining o'zgarish darajasi yetarli darajada katta bo'lgandagina natija maqsadli bo'lsa boshqasi uchun beqaror natijani beradi. Bu degani biz amallarimizdan biror nimani o'zgartirmas ekanmiz yaxshi natijaga erisha olmaymiz. Har bir parameter uchun alohida o'rgatishni amalga oshirishimiz ham mumkin lekin juda ko'p parameterga ega bo'lgan modellar uchun bu ishni amalga oshirish oson kechmaydi.

Vaziyatni nazorat qilishning osonroq yo'li bor: Kiritilayotgan namunalarni shunday o'zgartiraylikki natijada gradientlar bir biridan unchalik farq qilmasin. Boshqacharoq aytganda kiritilayotgan namunalarning chegara sohasini -1.0 va 1.0 oralig'ida yaqinroq ko'ritishga olib kelishimiz kerak. bizning holatimizda bu ishni *x* ni 0.1 ga ko'raytirish orqali amalga oshiramiz.

Agar kiritilayotgan x namunalarning qiymatlarini tekshiradigan bo'lsak ularning qiymati turli ko'ritishda ekaniga guvoh bo'lamiz. 
```python
# In [3]
x
# Out [3]
# (tensor([ 34.5520,  74.4541,  80.9875,   3.4582,  56.4779,  26.9816,  95.7942, 106.2283,  61.1694,   1.0895,   8.9626]),
```



Yuqoridagi *x* namunani o'rgatishdan oldin uni qiymatini 0.1 martaga o'shirib olamiz.

```python
# In [12]
x = x * 0.1
x
# Out [12]
# tensor([ 3.4552,  7.4454,  8.0987,  0.3458,  5.6478,  2.6982,  9.5794, 10.6228, 6.1169,  0.1090,  0.8963])
```

Va o'rgatishni qayta takrorlaymiz.

```python
# In [12]
params = training_loop(
    n_epochs = 100,
    learning_rate = 1e-2,
    params = torch.tensor([1.0, 0.0]),
    x = x,
    y = y)
# Out [12]
# Epoch 1, Loss 0.622568
# 	 Params:  tensor([ 0.9400, -0.0039])
# 	 Grad:  tensor([5.9979, 0.3906])
# Epoch 2, Loss 0.399406
# 	 Params:  tensor([ 0.9257, -0.0017])
# 	 Grad:  tensor([ 1.4320, -0.2172])
# Epoch 3, Loss 0.385859
# 	 Params:  tensor([0.9220, 0.0018])
# 	 Grad:  tensor([ 0.3730, -0.3561])

#      ...

# Epoch 98, Loss 0.297541
# 	 Params:  tensor([0.8836, 0.2817])
# 	 Grad:  tensor([ 0.0284, -0.2121])
# Epoch 99, Loss 0.297085
# 	 Params:  tensor([0.8833, 0.2838])
# 	 Grad:  tensor([ 0.0282, -0.2107])
# Epoch 100, Loss 0.296634
# 	 Params:  tensor([0.8830, 0.2859])
# 	 Grad:  tensor([ 0.0280, -0.2093])

# tensor([0.8830, 0.2859])
```

Xatto o'rganish ko'rsatgichini (*learning_rate*) 1e-2ga o'zgartirganimizda ham portlash yuz bermadi. 

Avvalo gradientlarga qaraylik: *weight* va *bies*ning gradientlari bir hil o'lchamda ekaniga guvoh bo'lyapmiz yani ikkalasi uchun ha bir hil *learning_rate* ishlatish kifoya ekan. 

Agar *loss*ga etibor beradigan bo'lsak *loss*ning qiymati ancha minimallashganini ko'ramiz.

Me'yorlash (normalizing)ni faqatgina qiymatlarni 10ga kamaytirish bilangina emas boshqa amallar bilan yana ham yaxshiroq amalga oshirishimiz mumkin edi, lekin hozircha natija bizni qoniqtiradi va biz shu holda davom etamiz.

Keling takrorlanishlar sonini yetarli darajada oshirish orqali parameterlarimiz qanchalik minimallashishini ko'raylik. *Epoc*lar sonini 5000ga oshiramiz (*n_epochs = 5000*):

```python
# In [12]
params = training_loop(
    n_epochs = 5000,
    learning_rate = 1e-2,
    params = torch.tensor([1.0, 0.0]),
    x = x,
    y = y)
# Out [12]
# Epoch 1, Loss 0.622568
# 	 Params:  tensor([ 0.9400, -0.0039])
# 	 Grad:  tensor([5.9979, 0.3906])
# Epoch 2, Loss 0.399406
# 	 Params:  tensor([ 0.9257, -0.0017])
# 	 Grad:  tensor([ 1.4320, -0.2172])
# Epoch 3, Loss 0.385859
# 	 Params:  tensor([0.9220, 0.0018])
# 	 Grad:  tensor([ 0.3730, -0.3561])

#       ...

# Epoch 4998, Loss 0.262982
# 	 Params:  tensor([0.8410, 0.5997])
# 	 Grad:  tensor([ 2.3842e-06, -2.7493e-06])
# Epoch 4999, Loss 0.262982
# 	 Params:  tensor([0.8410, 0.5997])
# 	 Grad:  tensor([ 2.3842e-06, -2.7493e-06])
# Epoch 5000, Loss 0.262982
# 	 Params:  tensor([0.8410, 0.5997])
# 	 Grad:  tensor([ 2.3842e-06, -2.7493e-06])

# tensor([0.8410, 0.5997])
```

Juda soz. Gradientning kamayish tomonga parameterlarni yangilab borganimizsari *loss*ning qiymati ham pasayib bordi. *loss*ning qiymati nolga yaqinlashyapti lekin aniq nolga teng emas. Bu takrorlanishlar yetarli bo'lmaganini anglatishi mumkin yoki ma'lumot(data)ning nuqtalari aniq bir chiziqda yotmasligini anglatadi. Kutganimizdek bizning ma'lumotlarimiz mukammal aniqlikka ega emas, yoki ma'lumotlarni o'qishda ba'zi xatoliklarga yo'l qo'yilgan. 

#### ESLATMA
> Bu yerda me'yorlash modelni o'rgatishda ancha yordam berdi, lekin bunaqa model uchun parameterlarni meyorlashga xojat yo'q deb aytishingiz mumkin. Bu mutlaqo to'g'ri. Yuqoridagi muammo parameterlar bilan ishlashda bir qancha yo'llaridan foydalanib yechish mumkin bo'lgan darajada kichik muammodir.
> Lekin, katta va ancha murakkab muammolarda modelning optimalligi oshirishda me'yorlash (normalization) oson va samarali (albatta, hali qiluvchuv emas) vosita hisoblanadi.

<br/>

## **2-QADAM:** Natijani tasvirda ifodalash

Keling endi ma'lumotlarni tasvirda ko'ramiz. Bu *data  science* uchun har kim qilib ko'rishi kerak bo'lgan birinchi ish bo'lishi kerak. Har doim ma'lumotlarni tasvirda ko'rishga xarakat qiling:

```python
%matplotlib inline
from matplotlib import pyplot as plt
yy = model(x, *params)
fig = plt.figure(dpi=600)
plt.xlabel("x label")
plt.ylabel("y label")
plt.plot(x.numpy(), yy.detach().numpy())
plt.plot(x.numpy(), y.numpy(), 'o')
plt.savefig("dl_ch1t2_plot01.png", format="png")
```

Yuqorida *`*params`* shaklida parameterlarimizni modelga uzatdik. parameterlarni bunday shaklda uzatish Pythonda *argument unpacking* deb nomlanadi. Python dasturlash tilida *list* yoki *tuple*lar bilan odatda shunday ishlanadi. Lekin PyTorchda ham tensorni ham shunday uzatish mumkin, shunda tensor o'lcham bo'ylab yoyiladi. Hullas bu yerda ***model(x, *params)*** ***model(x, params[0], params[1])***ga tengdir.

![data_plot](https://martianvenusian.github.io/dl_tutorial/codes/tutorial_1/dl_ch1t2_plot01.png)

Tasvirdan hulosa qiladigan bo'lsak, ma'lumotlarimiz uchun chiqizli model eng maqbul madelga o'xshaydi.