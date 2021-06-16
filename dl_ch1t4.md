# Gradientni avtamatik hisoblash

<br/>

### Bu bo'lim nimani o'rgatadi
 *  

### Hal qilinishi kerak bo'lgan muammo
 *  

<br/>

* * *

<br/>

Biz o'tgan darslarda optimallas uchun oddiy holatlar uchun unchalik yomon natija bermaydigan *vanilla* gradient descentdan foydalandik. Ortiqcha gapga xojat yo'q, optimallash usullarining bir necha stratigiya va ayyorlik yo'llari bor va bu ayniqcha murakkab modellarda yordam beradi.

Hozircha dasturchi kodidan uzoqlashtiradigan PyTorchning optimallash yo'llaridan biri haqida gaplashamiz (misol uchun o'tgan darslarda ko'rgan o'rgatishni takrorlovchi qism). Bu har safar har bir parameterni yangilashda to'g'ridan-to'g'ri ishtirok etishimizdan ozoq qiladi. *torch* bo'limi *optim* bo'limga ega va biz o'sha yerda turli optimallash algoritmlarini qo'llaydigan *class*larni topishimiz mumkin:

```python
# In [6]
import torch.optim as optim
dir(optim)

# Out [6]
# ['ASGD',
#  'Adadelta',
#  'Adagrad',
#  'Adam',
#  'AdamW',
#  'Adamax',
#  'LBFGS',
#  'Optimizer',
#  'RMSprop',
#  'Rprop',
#  'SGD',
#  'SparseAdam',
#  '__builtins__',
#  '__cached__',
#  '__doc__',
#  '__file__',
#  '__loader__',
#  '__name__',
#  '__package__',
#  '__path__',
#  '__spec__',
#  '_multi_tensor',
#  'functional',
#  'lr_scheduler',
#  'swa_utils']
```
Har bir optimizerlarning konstruktorlari birinchi kiritiladigan parameter sifatida parameterlar listini qabul qiladi (PyTorch tensorlari uchun *requires_grad* odatda *True* bo'ladi). Optimizer kiritilgan barcha parameterlar optimizerning ichida qayta o'rgatiladi va shundagina optimizer ularning qiymatlarini yangilay oladi va *grad* atributiga murojaat qila oladi.

Har bir optimizer ikkita metodni ochib beradi: *zero_grad* va *step*. *zero_grad* konstraktor yordamida optimizerga kiritilgan barcha parameterlarning *grad* atributini nolgan o'zlashtiradi. *step* esa maxsus optimizer yordamida qo'llangan optimallash strategiyasiga ko'ra shu parameterlarning qiymatlarini yangilaydi.

#### GRADIENT DESCENT OPTIMIZERni ishlatish.

Keling parameterlar yaratib gradient descent optimizerni hosil qilaylik:

```python
# In [7]
params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-5
optimizer = optim.SGD([params], lr=learning_rate)
```

Bu yerda SGD *stochastic gradient descent*ning qisqartmasidir. Aslida, bu optimizerning o'zi ham aniq *vanilla* gradient descent bo'ladi (agar *momentum* argumenti 0.0 teng bo'lsa va albatta boshlang'ich *momentum* doim 0.0 bo'ladi). *stochastic* atamasi barcha kiritilgan namunalarning tasodifiy to'plamlarining o'rtacha qiymatini hisoblash natijasida olingan gradientdan kelib chiqqan faktdga nisbatan ishlatiladi. Biroq, *loss* barcha namunalari(vanilla)dan foydalanib hisoblanganmi yoki bu namunalarning tasodifiy to'plamlari(stochastic)dan foydalanib hisoblanganmi buni optimizer bilmaydi. Shuining uchun ikki holatda ham algoritm bir hildir:

```python
# In [8]
yy = model(x, *params)
loss = loss_fn(yy, y)
loss.backward()
optimizer.step()

params
# Out [8]
# tensor([ 9.3147e-01, -9.0416e-04], requires_grad=True)
```

Bizning aralashuvimizsiz ham *step*ni chaqirish orqali parameterlarning qiymatlari yangilandi. Gap shundaki, *optimizer* *params.grad*ga murojat qiladi va *params*ni yangilaydi ya'ni, *learning_rate* marta *grad*ni undan ayirib tashlaydi, huddi biz o'tgan darslarda qilganimiz kabi.

Ho'sh bu kodni o'rgatishni takrorlovchi qisimga qo'shamizmi? Yo'q! Biz gradientlarni nolga teglashni unutdik. Agar o'tgan darsdagi o'rgatishni takrorlovchi qismga etibor beradigan bo'lsak, har bir *backward* chaqirilganda gradientlar zanjir togunlarida yig'ilgan edi va bizning gradient descentimiz har yerda yoyilib turgandi! 

Mana bu yerda *zero_grad* to'g'ri joyga qo'yilgan (*backward*ni chaqirishdan shundoq oldin) o'rgatishni takrorlovchi tayyor kod:

```python
# In [9]
x_u = x * 0.1
params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)
yy = model(x_u, *params)
loss = loss_fn(yy, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
params
# out [9]
# tensor([ 0.9400, -0.0039], requires_grad=True)
```
Juda soz!. Ko'ryapsizmi *optim* moduli maxsus optimallashtirish sxemasini qo'llashdan ozod bo'lishimizga qanchalar yordam beryapti. Bizning yagona qiladigan ishimiz unga parameterlar listni taqdim qilishdan iborat (bu list nihoyat darajada uzun bo'lishi ham mumkin. Odatda neural network modellari uchun shunday listlar shunday uzun bo'ladi) va uning mayda tafsilotlari haqida qayg'urishimizga xojat yo'q

Keling endi o'rgatishni takrorlovchi qimsni yangilab olamiz:

```python
# In [10]
def training_loop(n_epochs, optimizer, params, x, y):
    for epoch in range(1, n_epochs + 1):
        yy = model(x_u, *params)
        loss = loss_fn(yy, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
    return params
```

```python
# In [11]
params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)

training_loop(
    n_epochs = 5000,
    optimizer = optimizer,
    params = params,
    x = x_u,
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

# Out [11]
# tensor([0.8410, 0.5997], requires_grad=True)
```

Ko'rib turganingizdek o'tgan darslarniki kabi bir xil natija oldik.

#### Boshqa OPTIMIZERlarni ishlatib ko'rish

Yana ham ko'proq optimizerlarni test qilib ko'rish uchun, bizni yagona qiladigan ishimiz boshqa optimizerlarni o'zlashtirish yetarli, shu jumladan *SGD*ning o'rniga *Adam*ni. Kodning qolgan qismlari o'z holida qoladi. Bu bizga anchagina qulaylik tug'diradi.

*Adam* haqida ko'p gapirib o'tirmasdan lo'nda qilib aytadigan bo'lsak o'rganishlar ko'rsatgichini (learning rate) moslashuvchan tarzda o'rnatiladigan ancha izchil bo'lgan optimizerder. Qo'shimcha qiladigan bo'lsak u parameterlarning ko'rsatgichlariga nisbatan ko'p ham injiq emas - parameterlarning ko'rsatgichlariga shu darajada befarqki xatto biz kiritilayotgan *x*ning qiymatini me'yorlamasdan (non-normalized) o'rganish ko'rsatgichini 1e-1 gacha oshirishimiz mumkin va Adam xatto qosh ham qoqmaydi.

```python
# In [12]
params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-1
optimizer = optim.Adam([params], lr=learning_rate)

training_loop(
    n_epochs = 5000,
    optimizer = optimizer,
    params = params,
    x = x,
    y = y)

# Epoch 500, Loss 0.265097
# Epoch 1000, Loss 0.262982
# Epoch 1500, Loss 0.262982
# Epoch 2000, Loss 0.262982
# Epoch 2500, Loss 0.262983
# Epoch 3000, Loss 0.262982
# Epoch 3500, Loss 0.262982
# Epoch 4000, Loss 0.262982
# Epoch 4500, Loss 0.262982
# Epoch 5000, Loss 0.262982

# Out [12]
# tensor([0.0841, 0.5997], requires_grad=True)
```