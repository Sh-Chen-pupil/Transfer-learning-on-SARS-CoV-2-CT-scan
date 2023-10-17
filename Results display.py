from keras.preprocessing import image
import os
img = image.load_img('./data/validation/COVID-positive/Covid (216).png', grayscale=False, target_size=(64, 64))
show_img=image.load_img('./data/validation/COVID-positive/Covid (216).png', grayscale=False, target_size=(200, 200))
disease_class=['Covid-19','Non Covid-19']
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
x /= 255

custom = model.predict(x)
print(custom[0])
plt.subplot(2,2,1)
plt.imshow(show_img)
plt.title('fact:Covid')

a=custom[0]
ind=np.argmax(a)
        
print('(1,1)Prediction:',disease_class[ind])

img = image.load_img('./data/validation/COVID-positive/Covid (984).png', grayscale=False, target_size=(64, 64))
show_img=image.load_img('./data/validation/COVID-positive/Covid (984).png', grayscale=False, target_size=(200, 200))
disease_class=['Covid-19','Non Covid-19']
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
x /= 255

custom = model.predict(x)
print(custom[0])
plt.subplot(2,2,2)
plt.imshow(show_img)
plt.title('fact:Covid')

a=custom[0]
ind=np.argmax(a)
        
print('(1，2)Prediction:',disease_class[ind])
img = image.load_img('./data/validation/COVID-negative/Non-Covid (1029).png', grayscale=False, target_size=(64, 64))
show_img=image.load_img('./data/validation/COVID-negative/Non-Covid (1029).png', grayscale=False, target_size=(200, 200))
disease_class=['Covid-19','Non Covid-19']
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
x /= 255

custom = model.predict(x)
print(custom[0])
plt.subplot(2,2,3)
plt.imshow(show_img)
plt.title('fact:Non-Covid')

a=custom[0]
ind=np.argmax(a)
        
print('(2,1)Prediction:',disease_class[ind])

img = image.load_img('./data/validation/COVID-negative/Non-Covid (102).png', grayscale=False, target_size=(64, 64))
show_img=image.load_img('./data/validation/COVID-negative/Non-Covid (102).png', grayscale=False, target_size=(200, 200))
disease_class=['Covid-19','Non Covid-19']
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
x /= 255

custom = model.predict(x)
print(custom[0])

a=custom[0]
ind=np.argmax(a)
        
print('(2,2)Prediction:',disease_class[ind])
plt.subplot(2,2,4)
plt.imshow(show_img)
plt.title('fact:Non-Covid')
plt.show()
