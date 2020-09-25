

import torch

# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:')

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# In[2]:


import pandas as pd

trans = pd.read_csv("DATA/transaction_final.csv")
product=pd.read_csv("DATA/products_final.csv")


trans_products=trans.merge(product,on='PRODUCT_ID',how='inner')
trans_products=trans_products[trans_products['SUB_COMMODITY_DESC']!='GASOLINE-REG UNLEADED']



# Report the number of transactions & products.
print('Number of transactions : {:,}\n'.format(trans.shape[0]))
print('Number of products: {:,}\n'.format(product.shape[0]))


# Display 10 random rows from the data.
print(trans.sample(10))
print(product.tail(10))


# In[3]:


assortments=trans_products.groupby(['STORE_ID','WEEK_NO','DEPARTMENT']).agg(lambda x:list(x)).reset_index()
assortments['SIZE']=assortments['PRODUCT_ID'].apply(lambda x:len(x))


# In[4]:


labels = assortments.WEEK_QUANTITY.values
assortment = assortments.INDEX_x.values


# In[9]:


assortments[assortments['SIZE']>400]


# In[10]:



# Print the sample Assortment.
print(' Assortment: ', assortment[7])

# Print the corresponding label set.
print('Labels: ', labels[7])


# In[13]:


training_loss=[]
training_time=[]
validation_loss=[]
validation_time=[]


# In[31]:


assortment_sizes=[32,40,64,80,128,160,256,300,320,400]


# In[51]:


from keras.preprocessing.sequence import pad_sequences
for i in assortment_sizes:
    MAX_LEN = i

    print('\nPadding/truncating all assortments to %d values...' % MAX_LEN)
    product_tokens= pad_sequences(assortment, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    demand_labels= pad_sequences(labels, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    print('\nDone.')


    # In[52]:


    attention_masks = []

    # For each assortment...
    for sent in product_tokens:

        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for the assortment.
        attention_masks.append(att_mask)


    # In[53]:




    from sklearn.model_selection import train_test_split
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(product_tokens, demand_labels, 
                                                                random_state=2018, test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,
                                                 random_state=2018, test_size=0.1)


    # In[54]:


    import torch
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)

    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)

    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)


    # In[55]:


    from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
    batch_size =64

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


    # In[56]:


    import sys
    sys.path.insert(1, 'pro-bert')
    from modeling_bert import ProductBert,BertConfig

    import torch

    configuration = BertConfig()


    model = ProductBert(configuration)


    # In[57]:


    output=model(torch.tensor(assortment[12434]).unsqueeze(0), 
                        token_type_ids=None, 
                        attention_mask=None, 
                        labels=torch.tensor(labels[12434]).unsqueeze(0))


    # In[58]:


    model.cuda()


    # In[59]:


    model.parameters()


    # In[60]:


    params = list(model.named_parameters())
    params = list(model.named_parameters())

    print('The BERT model has {:} different named parameters.\n'.format(len(params)))

    print('==== Embedding Layer ====\n')

    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== First Transformer ====\n')

    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== Output Layer ====\n')

    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


    # In[61]:


    from transformers import  AdamW
    import torch.optim
    optimizer=torch.optim.SGD(model.parameters(), lr=5e-3)
    # optimizer = AdamW(model.parameters(),
    #                   lr = 5e-3, # args.learning_rate - default is 5e-5, our notebook had 2e-5
    #                   eps = 1e-6 # args.adam_epsilon  - default is 1e-8.
    #                 )
    optimizer=torch.optim.Adam(model.parameters(),lr=0.005, betas=(0.9, 0.999), eps=1e-06, weight_decay=0, amsgrad=False)


    # In[62]:



    from transformers import get_linear_schedule_with_warmup

    # Number of training epochs (authors recommend between 2 and 4)
    epochs = 15

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)


    # In[63]:


    import numpy as np


    # In[64]:


    import time
    import datetime

    def format_time(elapsed):
       
        elapsed_rounded = int(round((elapsed)))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))


    # In[65]:


    import random
    import time
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    # model.to(device)

    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        model.train()


        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

          
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)


            model.zero_grad()        

            outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)
   

            loss = outputs[0]
            if step%1==0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}. with Loss: {:} with total_loss: {:}'.format(step, len(train_dataloader), elapsed,loss.item(),total_loss))
  
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()


    #         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # Update the learning rate.
            scheduler.step()
    #         
            # Report progress.

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)  


        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

    training_loss.append(avg_train_loss)
    training_time.append(elapsed) 
    print("")
    print("Training complete!")


    # In[66]:


    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    total=0
    for batch in validation_dataloader:

        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():        

            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask,labels=b_labels)

       
        logits = outputs
        print(logits[1].view(-1))
        print(b_labels.view(-1))
        print('  Batch {:>5,}  of  {:>5,}. with Loss: {:} '.format(nb_eval_steps, len(validation_dataloader),logits[0].item()))

        total+=logits[0].item()

     
        nb_eval_steps += 1

    # Report the final accuracy for this validation run.
    print("  Loss: {0:.2f}".format(total/nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))
    validation_loss.append(total/nb_eval_steps)
    validation_time.append(format_time(time.time() - t0))


    # In[67]:


    print("training loss")
    print(training_loss)
    print("training time")
    print(training_time)
    print("validation loss")
    print(validation_loss)
    print("validation time")
    print(validation_time)


    # In[ ]:




