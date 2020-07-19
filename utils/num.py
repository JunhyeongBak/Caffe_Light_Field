for txt_i in range(25):
    f = open('/docker/lf_depth/datas/FlowerLF/source'+str(txt_i)+'.txt', 'w')
    for i in range(2792):
        txt_i2 = None
        if txt_i <= 4:
            txt_i2 = txt_i + 9
        elif txt_i > 4 and txt_i <= 9:
            txt_i2 =  txt_i + 12
        elif txt_i > 9 and txt_i <= 14:
            txt_i2 =  txt_i + 15
        elif txt_i > 14 and txt_i <= 19:
            txt_i2 =  txt_i + 18
        elif txt_i > 19 and txt_i <= 24:
            txt_i2 =  txt_i + 21
        f.write('/docker/lf_depth/datas/FlowerLF/'+str(txt_i2)+'/'+str(i)+'.png'+' 0\n')
    f.close()