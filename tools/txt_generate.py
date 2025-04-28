import os

def txt_generate(root,img_folder,label_folder,output_file):
    imgpath=os.path.join(root,img_folder)
    img_files = os.listdir(imgpath)
    output_file=root+"/"+output_file
    with open(output_file, 'w') as f:
        for img_file in img_files:
            img_name=os.path.splitext(img_file)[0]
            if os.path.exists(root+label_folder+img_name+'.png'):
                img_path=img_folder+img_file
                label_path=label_folder+img_name+'.png'
                f.write(f"{img_path} {label_path}\n")

if __name__ == '__main__':
    root="D:\data\datasets\Deepcrack/"
    img_folder="train_img/"
    label_folder="train_lab/"
    output_file="train.txt"
    txt_generate(root,img_folder,label_folder,output_file)