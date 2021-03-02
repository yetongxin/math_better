# coding: utf-8
from src.train_and_evaluate import *
from src.models import *
import time
import torch.optim
from src.expressions_transfer import *
from src.global_vars import set_input_lang, set_output_lang, device
import os
from src.pre_data import to_one_hot

if not os.path.exists('error'):
    os.makedirs('error')
if not os.path.exists('models'):
    os.makedirs('models')
batch_size = 64
embedding_size = 512 # todo: 128 linear->512
hidden_size = 512
n_epochs = 80
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 5
n_layers = 2

data = load_raw_data("data/math23k-processed.json")

pairs, generate_nums, copy_nums = transfer_num(data)

temp_pairs = []
for p in pairs:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3], p[4]))
pairs = temp_pairs

fold_size = int(len(pairs) * 0.2)
fold_pairs = []
for split_fold in range(4):
    fold_start = fold_size * split_fold
    fold_end = fold_size * (split_fold + 1)
    fold_pairs.append(pairs[fold_start:fold_end])
fold_pairs.append(pairs[(fold_size * 4):])

best_acc_fold = []

def parse_num_pos(num_pos, seq_len, max_len):
    arr = [0 for _ in range(max_len)]
    for pos in num_pos:
        arr[pos] = 1
    return arr

def writeFile(arr, fold, times):
  filename = 'error/' + str(fold) + '-' + str(times) + '.txt'
  print('write error file:', filename)
  f = open(filename, 'wt')
  f.writelines(arr)
  f.close()

# test: 从第2个fold开始计算
for fold in range(0, 5):
    pairs_tested = []
    pairs_trained = []
    for fold_t in range(5):
        if fold_t == fold:
            pairs_tested += fold_pairs[fold_t]
        else:
            pairs_trained += fold_pairs[fold_t]

    print('pair_tested len:', len(pairs_tested), pairs_tested[0])
    input_lang, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 5, generate_nums,
                                                                    copy_nums, tree=True)
    print(test_pairs[0], 'flag:', test_pairs[0][7], 'one hot value:', to_one_hot(test_pairs[0][7], len(test_pairs[0][7]),  len(test_pairs[0][7])),
          'input shape', torch.LongTensor(test_pairs[0][0]).unsqueeze(1).shape,
          'onehot:', torch.Tensor(to_one_hot(test_pairs[0][7], len(test_pairs[0][7]),  len(test_pairs[0][7]))).unsqueeze(1).shape,
          'second pair',test_pairs[1])
    set_input_lang(input_lang)
    set_output_lang(output_lang)
    # Initialize models
    encoder = EncoderSeq4(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                         n_layers=n_layers)
    # encoder = EncoderRNNAttn(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
    #                          n_layers=n_layers, dropout=0.5, d_ff=2048, N=1)
    predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                         input_size=len(generate_nums))
    generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                            embedding_size=embedding_size)
    merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
    # the embedding layer is  only for generated number embeddings, operators, and paddings

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
    merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)

    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=20, gamma=0.5)
    predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=20, gamma=0.5)
    generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=20, gamma=0.5)
    merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=20, gamma=0.5)

    # Move models to GPU
    if USE_CUDA:
        encoder = encoder.to(device)
        predict = predict.to(device)
        generate = generate.to(device)
        merge = merge.to(device)

    generate_num_ids = []
    for num in generate_nums:
        generate_num_ids.append(output_lang.word2index[num])

    evalate_times = 0
    pre_accuracy = 0
    best_accuracy = 0
    for epoch in range(n_epochs):
        encoder_scheduler.step()
        predict_scheduler.step()
        generate_scheduler.step()
        merge_scheduler.step()
        loss_total = 0
        input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, flag_batches, parsed_num_pos_batches = prepare_train_batch(train_pairs, batch_size)
        print("fold:", fold + 1)
        print("epoch:", epoch + 1)
        start = time.time()
        # print('input lengths', len(input_lengths), input_lengths)

        for idx in range(len(input_lengths)):
            # print('num_pos :', num_pos_batches[idx][0], parsed_num_pos_batches[idx][0], input_lang.index2string(input_batches[idx][0]))
            loss = train_tree(
                input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
                num_stack_batches[idx], num_size_batches[idx], generate_num_ids, encoder, predict, generate, merge,
                encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, output_lang, num_pos_batches[idx], flag_batch=flag_batches[idx],
                parsed_num_pos_batch=parsed_num_pos_batches[idx])
            loss_total += loss

        avg_loss = loss_total / len(input_lengths)
        print("loss:", avg_loss)
        print("training time", time_since(time.time() - start))
        print("--------------------------------")
        # 0.0452444252413537
        if epoch % 10 == 0 or epoch > n_epochs - 5 or best_accuracy > 0.743:
            evalate_times += 1
            value_ac = 0
            equation_ac = 0
            eval_total = 0
            start = time.time()
            error_list = []
            for test_batch in test_pairs:
                # test_batch: (input_sentence_index, len(input_sentecnce), output_sentence_index, len(output_sentence), nums, num_pos, num_stack, flag, num_pos)
                # print('evaluate tree', test_batch[7], to_one_hot(test_batch[7], len(test_batch[7]), len(test_batch[7])))
                test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                         merge, output_lang, test_batch[5], flag_batch=to_one_hot(test_batch[7], len(test_batch[7]), len(test_batch[7])),
                                         parsed_num_pos_batch=parse_num_pos(test_batch[5], 0, test_batch[1]), beam_size=beam_size)
                val_ac, equ_ac, gen_res, tar_res = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
                if val_ac:
                    value_ac += 1
                if equ_ac:
                    equation_ac += 1
                if val_ac == False and equ_ac == False:
                  tmp = input_lang.index2string(test_batch[0])
                  error_list.append(tmp + ' '.join(map(str, gen_res)) + '  real:' + ' '.join(map(str, tar_res)) + '\n')
                eval_total += 1
            pre_accuracy = float(value_ac) / eval_total
            if pre_accuracy > best_accuracy:
                best_accuracy = pre_accuracy
            print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
            print("testing time", time_since(time.time() - start))
            print("------------------------------------------------------")
            writeFile(error_list, fold, evalate_times)
            torch.save(encoder.state_dict(), "models/encoder")
            torch.save(predict.state_dict(), "models/predict")
            torch.save(generate.state_dict(), "models/generate")
            torch.save(merge.state_dict(), "models/merge")
            if epoch == n_epochs - 1:
                best_acc_fold.append((equation_ac, value_ac, eval_total))

a, b, c = 0, 0, 0
for bl in range(len(best_acc_fold)):
    a += best_acc_fold[bl][0]
    b += best_acc_fold[bl][1]
    c += best_acc_fold[bl][2]
    print(best_acc_fold[bl])
print(a / float(c), b / float(c))