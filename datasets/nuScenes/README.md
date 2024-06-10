# NuScenes

The dataset is large (395GB), hence the download may take considerable time. It is advisable to run the script with `nohup` and putting the args inside `''` (to avoid potential problems with escape characters):

```bash
nohup python get_files_nuScenes.py 'user_email' 'user_password'
```

To obtain `user_email` and `user_password`, one must register at [nuscenes.org](https://www.nuscenes.org/nuscenes)