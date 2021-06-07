## Tutorial 1
## Bu bo'lim nimalarni o'rgatadi
 * Ma'lumot (data)lar bilan ishlash va ularni o'rganish
 * Madelni to'g'ri tanlash 
 * Sodda algoritm yordamida muammoni o'rganib chiqish 
 * PyTorch yordamida bu algoritmni dasturlash
 * `weight`, `bieas`, `loss function` haqida tushunchalar

## Hal qilinishi kerak bo'lgan muammo
 * Muammoning mohiyati va berilgan ma'lumot(data)larga qarab va madelni to'g'ri tanlash.
 * Tanlangan model parameterlarni shunday taxmin qilsinki toki yangi ma'lumotlar berilganda ham model yaxshi natijalarni bera oladigan bo'lsin.

### **1-QADAM:** Ma'lumotlarni o'rganish

 - Aytaylik, bizda ma'lum bir miqdorlarni ko'rsatadigan ***x*** va ***y*** ma'lumotlari bor. 

```python
y = [4.209015  , 6.0251656 , 6.586659  , 1.0785204 , 5.323591,   2.9644287, 8.885769  , 9.895647  ,  6.464806  , 0.18034637, 1.2534696]
x = [34.552039 , 74.45411  , 80.987488 ,  3.458197 , 56.4778655, 26.98163  , 95.79415  , 106.228316 , 61.169422 , 1.089516 , 8.962632]
```

 - Ma'lumotlar bilan ishlashda ma'lumotlarni tensorga o'tkazib olishimiz kerak bo'ladi. Va PyTorch yordamida ma'lumotlarni tensorda ifodalash quydagicha amalgan oshiriladi.
```python
t_y = torch.tensor(y)
t_x = torch.tensor(x)
```

 - Bu ma'lumotlarni tasvirda ko'ramiz. Bu esa ma'lumotlarni yana ham  yaxshiroq o'rganishizga yordam beradi.
```python
fig = plt.figure(dpi=500)
plt.xlabel("x label")
plt.ylabel("y label")
plt.plot(t_x.numpy(), t_y.numpy(), 'o')
plt.savefig("temp_data_plot.png", format="png")
```

![Octocat](https://github.martianvenusian.io/dl_tutorial/blob/main/codes/tutorial_1/temp_data_plot.png)

### **2-QADAM:** Modelni tanlash
 - Ortiqcha izlanishlarsiz, yuqoridagi tasvirdan kelib chiqib bu muammoning yechimi ikki o'lchamli sodda model yotganini bilib olsak bo'ladi. Va *x* va *y* ma'lumotlar bir biri bilan chiziqli bog'liqlikka ega deb taxmin qilgan holda quydagi modelni tanlaymiz:

### y = w * x + b

 - Biz [`weight`]() va [`bias`]()dan kelib chiqib ***w*** va ***b*** haftlarini modelimiz uchun belgilab oldik. Bu ikkala atama chiziqli masshtablash (linear scaling) va o'zgarmas qo'shimchalar (additive constant)lar uchun odatiy atama bo'lib, bundan keyin biz bu atamalarga qayta qayta to'qnash kelamiz.

 - Juda soz. Galdagi vazifamiz bizdagi ma'lumot(data)lardan kelib chiqb bu ikki ***w*** va ***b***larni hisoblab chiqishimiz kerak bo'ladi. Buni shuning uchun qilishimiz kerakki, modelimiz orqali ***x*** ma'lumotlarimiz yordamida hosil qilinadigan yangi ***yy*** ma'lumotlar bizda mavjud bo'lgan ***y*** ma'lumotlar bilan yaqin bo'lsin. Bu huddi ma'lumotlar deyarli bir to'g'ri chiziqda yotadi deganidir. Aynan biz shuni amalga oshirmoqchimiz.

 - Biz bu amallarni PyTorch yordamida amalga oshiramiz va [`neural network`]()ni [`training`]() qilish mohiyatan modeni uning bir nechta parameterlaridan foydalanib biroz pishiqroq(aniqroq natija beradigan) modelga aylantirishni anglatadi.

 - Keling buni kengroq yoritaylik: Aytaylik bizda nomalum parameterlarga ega bo'lgan bir model bor, va shu parameterlarni shunday o'zgartiraylikki, oqibatda model bergan natija bilan bu natijani baholaydigan qiymatlarning orasidagi xatolik imkon qadar kichik bo'lsin. Bunday kelib chiqadiki model bergan natija va buni baholaydigan qiymatlarning orasidagi xatolikni topadigan o'ljov bizga kerak bo'ladi. Biz [`loss function`]() deb ataydigan ana shu o'lchov agar xatolik katta bo'lsa katta bo'lishi va agar kichik bo'lsa  mukammal moslik uchun kichik bo'lishi kerak. Bizning optimallashtirish ishlarimiz ***w*** va ***b***larni aniqlik bilan topishga qaratilgan bo'lishi kerakki, nitijada [`loss function`]() imkon qadar minumal bo'lsin.