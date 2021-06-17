# Gradientni avtamatik hisoblash

<br/>

### Bu bo'lim nimani o'rgatadi
 *  

### Hal qilinishi kerak bo'lgan muammo
 *  

<br/>

* * *

<br/>

Biz ma'lumotlarni (datalarni) model yordamida o'rgatishimizdan oldin, ma'lumotlarning bir qismini alohida ajratib olib qolgan qismini model yordamida o'rgatganimz ma'qul. Ajratib olingan ma'lumotlar *validation data* deb yuritiladi. Bu juda muhimdir, chunki siz modelingizni o'rgatganinggizdan keyin *validition data* yordamida bu modelni tekshirib u qanchalik ishonchli ekanini tasdiqlaysiz. Yuqori moslashuvchanlikka ega bo'lgan model berilgan ma'lumotlarga asosan *loss*ning qiymatini minimallashtirish uchun odatda juda ko'p parameterlardan foydalanishga moyil bo'ladi. Lekin, bu model yangi kiritilgan ma'lumotlarga nisbatan yaxshi natija berishiga kafolat bera olmaymiz. Tabiiyki, agar bizda *loss*ni hisoblashda yoki uning manfiy gradientini hisoblashda ishtirok etmagan mustaqil ma'lumotlar(datalar) bo'lsa va shu mustaqil ma'lumotlar yordamida *loss*ni tekshirib ko'radigan bo'lsak biz kutganimizdan ko'ra balandroq bo'lgan *loss*ga ega bo'lamiz. Biz bu hodisa haqida, ***overfitting*** haqida allaqachon aytib o'tganmiz.

Bizning bunga qarshi birinchi qila oladigan xarakatimiz bu *overffitng* hodisasi sodir bo'lishi mumkinligini payqashdan iborat.