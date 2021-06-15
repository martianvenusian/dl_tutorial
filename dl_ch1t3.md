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


### **1-QADAM:** Ma'lumotlarni me'yorlash