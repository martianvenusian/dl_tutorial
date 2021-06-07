## Tutorial 1
## Bu bo'lim nimalarni o'rgatadi
 * Algoritmlar ma'lumotlarni qanday o'rganadi
 * Sodda algoritm yordamida muammoni o'rganib chiqish
 * Qanday qilib datani tanlash, modelni tanlash va bu modelning parameterlarni shunday taxmin qilsinki toki yangi data berilganda ham yaxshi natijalarni bera oladigan bo'lsin.
 * PyTorch yordamida muammoni ko'rib chiqish

### *1-QADAM:* Ma'lumotlarni to'plash

 - Bizda haroratni o'lchovchi ikki o'lchov birligida ma'lumotlari bor. Biri *Celsus*da ba boshqasi bizga nomalum o'lchov birligida.

```python
t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]

t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
```

 - Bu ma'lumotlarni tensorga o'tkazib olamiz.
```python
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)
```

 - Bu ma'lumotlarni tasvirda ko'ramiz.
```python
fig = plt.figure(dpi=500)
plt.xlabel("Messurement")
plt.ylabel("Temperature (Celsuis)")
plt.plot(t_u.numpy(), t_c.numpy(), '0')
plt.savefig("temp_data_plot.png", format="png")
```