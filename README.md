Our work was published in 《IEEE International Conference on Multimedia and Expo (ICME)》 and 《IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)》 successively. We take the CNN model proposed in the journal as our final open source model. The code is implemented through pytorch.

In addition, we open source our own datasets. The dataset contains 177 sets of static scene exposure sequences taken by Cannon5D4 camera, each sequence contains the results of 7 shots, with an exposure interval of 1eV and ISO of 800. In order to advance the research in the field of raw images, all the original files after shooting are preserved.Specifically, it contains CR2 files, JPEG files, and detailed shooting parameters contained in the file ownership information. The training set, validation set and test set for this task are obtained by sampling the original files.

The download address and preprocessing code for the dataset can be downloaded from the following link:
link：https://pan.baidu.com/s/1tnY5I0zoCHesp2EFEEHg6w       Code for extraction：un57 

It is worth noting that the application domain of this dataset is not limited to single-shot HDRI tasks.

## Citation
If you find this work or code is helpful in your research, please cite:
```
@ARTICLE{9622212,
  author={Xu, Yilun and Liu, Ziyang and Wu, Xingming and Chen, Weihai and Wen, Changyun and Li, Zhengguo},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Deep Joint Demosaicing and High Dynamic Range Imaging within a Single Shot}, 
  year={2021},
  volume={},
  number={},
  pages={},
  doi={10.1109/TCSVT.2021.3129691}
}

@INPROCEEDINGS{9428198,
  author={Xu, Yilun and Liu, Ziyang and Wu, Xingming and Chen, Weihai and Li, Zhengguo},
  booktitle={2021 IEEE International Conference on Multimedia and Expo (ICME)}, 
  title={Restoration of HDR Images for SVE-Based HDRI via a Novel DCNN}, 
  year={2021},
  pages={1-6},
  doi={10.1109/ICME51207.2021.9428198}
}
```

## Contact
If you have any questions, feel free to E-mail me via: `yilunxu_buaa@163.com`

## License
The code is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License for Noncommercial use only. Any commercial use should get formal permission first.
