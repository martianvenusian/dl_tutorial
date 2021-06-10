# Tutorial 1: Model tanlash va Lossni hisoblash 
## Bu bo'lim nimalarni o'rgatadi
 * Ma'lumot (data)lar bilan ishlash va ularni o'rganish
 * Madelni to'g'ri tanlash 
 * Sodda algoritm yordamida muammoni o'rganib chiqish 
 * PyTorch yordamida algoritmni dasturlashga o'girish
 * `weight`, `bieas`, `loss function` haqida tushunchalar

## Hal qilinishi kerak bo'lgan muammo
 * Muammoning mohiyati va berilgan ma'lumot(data)larga qarab va madelni to'g'ri tanlash.
 * Tanlangan model parameterlarni shunday taxmin qilsinki toki yangi ma'lumotlar berilganda ham model yaxshi natijalarni bera oladigan bo'lsin.


* * *


<br/>

## **1-QADAM:** Ma'lumotlarni o'rganish

Aytaylik, bizda ma'lum bir miqdorlarni ko'rsatadigan ***x*** va ***y*** ma'lumotlari bor. 

```python
y = [4.209015  , 6.0251656 , 6.586659  , 1.0785204 , 5.323591,   2.9644287, 8.885769  , 9.895647  ,  6.464806  , 0.18034637, 1.2534696]
x = [34.552039 , 74.45411  , 80.987488 ,  3.458197 , 56.4778655, 26.98163  , 95.79415  , 106.228316 , 61.169422 , 1.089516 , 8.962632]
```

Ma'lumotlar bilan ishlashda ma'lumotlarni tensorga o'tkazib olishimiz kerak bo'ladi. Va PyTorch yordamida ma'lumotlarni tensorda ifodalash quydagicha amalgan oshiriladi.

```python
t_y = torch.tensor(y)
t_x = torch.tensor(x)
```

Bu ma'lumotlarni tasvirda ko'ramiz. Bu esa ma'lumotlarni yana ham  yaxshiroq o'rganishizga yordam beradi.

```python
fig = plt.figure(dpi=500)
plt.xlabel("x label")
plt.ylabel("y label")
plt.plot(t_x.numpy(), t_y.numpy(), 'o')
plt.savefig("temp_data_plot.png", format="png")
```

![data_plot](https://martianvenusian.github.io/dl_tutorial/images/temp_data_plot.png)

## **2-QADAM:** Modelni tanlash va lossni hisoblash
Ortiqcha izlanishlarsiz, yuqoridagi tasvirdan kelib chiqib bu muammoning yechimi ikki o'lchamli sodda model yotganini bilib olsak bo'ladi. Va *x* va *y* ma'lumotlar bir biri bilan chiziqli bog'liqlikka ega deb taxmin qilgan holda quydagi modelni tanlaymiz:

### y = w * x + b

Biz [`weight`] va [`bias`]dan kelib chiqib ***w*** va ***b*** haftlarini modelimiz uchun belgilab oldik. Bu ikkala atama chiziqli masshtablash (linear scaling) va o'zgarmas qo'shimchalar (additive constant)lar uchun odatiy atama bo'lib, bundan keyin biz bu atamalarga qayta qayta to'qnash kelamiz.

Juda soz. Galdagi vazifamiz bizdagi ma'lumot(data)lardan kelib chiqb bu ikki ***w*** va ***b***larni hisoblab chiqishimiz kerak bo'ladi. Buni shuning uchun qilishimiz kerakki, modelimiz orqali ***x*** ma'lumotlarimiz yordamida hosil qilinadigan yangi ***yy*** ma'lumotlar bizda mavjud bo'lgan ***y*** ma'lumotlar bilan yaqin bo'lsin. Bu huddi ma'lumotlar deyarli bir to'g'ri chiziqda yotadi deganidir. Aynan biz shuni amalga oshirmoqchimiz.

Biz bu amallarni PyTorch yordamida amalga oshiramiz va [`neural network`]ni [`training`] qilish mohiyatan modeni uning bir nechta parameterlaridan foydalanib biroz pishiqroq(aniqroq natija beradigan) modelga aylantirishni anglatadi.

Keling buni kengroq yoritaylik: Aytaylik bizda nomalum parameterlarga ega bo'lgan bir model bor, va shu parameterlarni shunday o'zgartiraylikki, oqibatda model bergan natija bilan bu natijani baholaydigan qiymatlarning orasidagi xatolik imkon qadar kichik bo'lsin. Bunday kelib chiqadiki model bergan natija va buni baholaydigan qiymatlarning orasidagi xatolikni topadigan o'ljov bizga kerak bo'ladi. Biz [`loss function`] deb ataydigan ana shu o'lchov agar xatolik katta bo'lsa katta bo'lishi va agar kichik bo'lsa  mukammal moslik uchun kichik bo'lishi kerak. Bizning optimallashtirish ishlarimiz ***w*** va ***b***larni aniqlik bilan topishga qaratilgan bo'lishi kerakki, nitijada [`loss function`] imkon qadar minumal bo'lsin.

[***Loss function***] - bu funksiya bo'lib oddiy bir raqamli qiymatni hisoblaydi va butun o'rgatish jarayoni aynan shu qiymatni minimallashtirishga qaratiladi. [***Lost***] qiymatni hisoblash model yordamida o'rgatiladigan namunalardan olingan taxminiy natija bilan haqiqiy namunalardan olingan natija o'rtasidagi farqni hisoblashdan iborat. Bizning holatimizda, bu farq ***yy - y*** ni hisoblashdan iboratdir.

Bizning maqsadimiz ***yy*** va ***y*** o'rtasidagi munosabatni o'rganish va shuning uchun ***Lost***ning qiymati xar doim musbat bo'lishiga lozim. Buda bizda bir nechta tanlov bor va eng yaxshi tanlov bu ***|yy - y|*** yoki ***|yy - y|^2***. 

Ikkala tanlov ham nolda aniq minimalga ega and taxminiy qiymat haqiqiy qiymatdan har ikki tarafga uzoqlashgan sari monotonik o'sadi. Chunki o'sish qiyaligi ham minimaldan monotonik uzoqlashadi. Ularning ikkalasi ham *qavariq* deb ataladi. Bizning model ham chiziqli bo'lgani sababli, ***w*** va ***b***larning funksiyasi sifatida ***lost*** ham qavariqdir. ***Lost*** model parameterlarining qavariq funksiyasi bo'lsan hollarda maxsus algoritmlar yordamida minimalni topishning juda samarali yo'llari mavjud. Lekin, bu yerdsa hozircha ancha sodda lekin umumiy bo'lgan usullardan foydalanib turamiz. Chunki buni biz qiziqayotgan ***neural network*** uchun qilamiz va bu yearda ***loss*** kiritiladigan na'munalarning qovariq funksiyasi bo'lmaydi.

Bizning ikkala ***|yy - y|*** yoki ***|yy - y|^2*** funksiyalarimiz orasida ikkinchi funksiya ancha samaraliroqdir. Chunki faqtni 2chi darajaga oshirish minimumga yaqinlashgan sari xatolarni yengilroq jazolaydi katta xatolarni esa kuchliroq jazolaydi. Odatda, ko'proq kichik xatolarga ega bo'lish bir nechta jiddiy xatolardan ko'ra yaxshiroqdir, va 2chi darajaga oshirilgan varqlar bu maqsadimizga muofiqdir.


Biz modelni tanladik va ***loss function*** haqida bilib oldik. Endi bu matematik tushincha va algoritmni PyTorch yordamida dasturlaymiz. 

Modelning funksiyasini yaratamiz:
```Python
def model(x, w, b):
    return w * x + b
```
*x*, *w* va *b*larni mos ravishda `input` tensor, `weight` parameter va `bias` parameterlardir. Bu yerda parameterlar PyTorchning skalarlaridir (`scalar`: nol-o'lchamli tensor). Bu funksiya o'z navbatida tensor qiymat qaytaradi.

Endi *loss* funksiyani yaratamiz:
```python
def loss_fn(yy, y):
    squared_diffs = (yy - y) ** 2
    return squread_diffs.mean()
```
Shu o'rinda takidlash joiski, biz mos-elementlar orasidagi farqni hisoblab ularni 2chi darajaga oshirdik, va oxirida barcha elementlarning o'rtacha qiymatini hisobladik. Bu hisoblangan *loss* funksiya ***mean square loss** deb ataladi.

Keyingi qiladigan ishimiz modelni ishga tushirishdir.
```python
w = torch.ones(())
b = torch.zeros(())

yy = model(x, w, b)
```

*Loss* hisoblaymiz:
```python
loss = loss_fn(y, yy)
```


## **3-QADAM:** Lossni optimallashtirish
Yuqoridagi bo'limda biz `model`ni va `loss`ni hisobladik. Va nihoyat biz misolning eng muhim qismiga keldik: Qanday qilib ***loss***ni minimalga olib boradigan ***w*** va ***b***ni hisoblaymiz

Mos parameterlarga nisbatan *Loss* funksiyani ***gradient descent*** algoritmi yordamida optimallashtiramiz. *Gradient descent* asligda juda sodda g'oyadir va millionlab parameterlarga ega katta *neural network* modellarida *gradient descent*ning darajasi ajablanarli darajada yuksaladi.

***Gradient descent*** g'oyasi bu har bir parameteriga nisbatan *loss*ning o'zgarish tezligini (o'zgarish darajasini) hisoblashdan iborat. Biz *w* va *b*ga kichik qiymatlarni berib `loss` o'z qo'shnisiga nisbatan qanchalik o'zgarganini hisoblash orqali *loss*ning o'zgarish tezligi (o'zgarish darajasi)ni taxmin qilishimiz mumkin.

```python
delta = 0.1
loss_rate_of_change_w = (loss_fn(model(x, w + delta, b), y) - loss_fn(model(x, w - delta, b), y)) / (2.0 * delta)
```