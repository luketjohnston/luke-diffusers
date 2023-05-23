import csv
from tqdm import tqdm 
import numpy as np
import PIL.Image
import torch 
from torchvision import transforms
from torch.utils.data import Dataset
from bucketmanager import BucketManager



f  = '/persist/ljohnston/datasets/may17_finetuning/ar_train.csv'

# NOTE: when using accelerate, we don't need to bother with world_size/global_rank etc.
# Accelerate internally splits indices by rank
class ARBucketDataset(Dataset):
    def __init__(self, f, batch_size, world_size, global_rank, tokenizer):
        print('Reading csv...')
        r = csv.DictReader(open(f))
        self.ds = list(r)
        self.batch_size = batch_size
        self.world_size = world_size
        self.global_rank = global_rank
        self.tokenizer = tokenizer
        
        res_map = {}
        print('Reading aspect ratios...')
        for i,d in enumerate(self.ds):
          res_map[i] = (d['width'],d['height'])
        
        print('Initializing bucket manager...')
        self.bm = BucketManager(res_map, bsz=self.batch_size)
        self.bm.gen_buckets()
        self.bm.assign_buckets()
        self.batch_indices = None # set in setup_for_new_epoch method below
        self.setup_for_new_epoch()


    def setup_for_new_epoch(self):
         self.bm.start_epoch(world_size=self.world_size, global_rank=self.global_rank)
         self.batch_indices = list(self.bm.generator())

    # returns number of batches, not number of datapoints
    def __len__(self):
        return len(self.batch_indices)

    def tokenize_captions(self, captions):
        chosen_captions = []
        for caption in captions:
            if isinstance(caption, str):
                chosen_captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                chosen_captions.append(random.choice(caption) if is_train else caption[0])
            else:
                print("FAILED ON CAPTION: ", caption)
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = self.tokenizer(
            chosen_captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    def __getitem__(self, index):
        indices,(width,height) = self.batch_indices[index]

        captions = [self.ds[i]['caption'] for i in indices]
        image_filenames = [self.ds[i]['path'] for i in indices]

        ims = []

        # TODO make functional?
        mytransforms1 = transforms.Compose([
                transforms.CenterCrop((height, width)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
                ])

        #mytransforms2 = transforms.Compose([
        #        transforms.ToTensor(),
        #        transforms.Normalize([0.5], [0.5]),
        #        ])

        for i,fname in enumerate(image_filenames):
            im = PIL.Image.open(fname).convert('RGB')
            if width / im.width  > height / im.height:
                im = im.resize((width, int(im.height * (width / im.width))))
            else:
                im = im.resize((int(im.width * (height / im.height)), height))
            #im.save(f'tmp/idx_{index}_{width}_{height}_{i}_nocrop.jpg')
            im = mytransforms1(im)
            #im.save(f'tmp/idx_{index}_{width}_{height}_{i}.jpg')
            #im = mytransforms2(im)
            ims.append(im)

        tokenized_captions = self.tokenize_captions(captions)

        #print(f'Returning ims with length: {len(ims)}, and shape of 0: {ims[0].shape}')
        #print(f'Returning tokenized_captions.shape {tokenized_captions.shape}')

        return {'pixel_values': ims,'captions': captions, 'input_ids': tokenized_captions }



if __name__ == '__main__':
    ds = ARBucketDataset(f, 16, 8, 0)
    
    dataloader = torch.utils.data.DataLoader(ds, num_workers=32)
    
    for x in tqdm(dataloader, total=len(ds)):
        pass

