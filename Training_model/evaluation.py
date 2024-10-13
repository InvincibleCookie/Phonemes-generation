#Function to give a overview of the phoneme error rate
def error_rate(timit, asr,phn = True):
  if phn:
    r = timit.split()
    h = asr.split()
  else:
    r = timit.split()
    h = asr.split()

  #costs will holds the costs, like in the Levenshtein distance algorithm
  costs = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]

  # backtrace will hold the operations we've done.
  # so we could later backtrace, like the WER algorithm requires us to.
  backtrace = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]


  OP_OK = 0
  OP_SUB = 1
  OP_INS = 2
  OP_DEL = 3
  # First column represents the case where we achieve zero
  # hypothesis words by deleting all reference words.
  for i in range(1, len(r)+1):
      costs[i][0] = 1*i
      backtrace[i][0] = OP_DEL

  # First row represents the case where we achieve the hypothesis
  # by inserting all hypothesis words into a zero-length reference.
  for j in range(1, len(h) + 1):
      costs[0][j] = 1 * j
      backtrace[0][j] = OP_INS

  # computation
  for i in range(1, len(r)+1):
    for j in range(1, len(h)+1):
      if r[i-1] == h[j-1]:
        costs[i][j] = costs[i-1][j-1]
        backtrace[i][j] = OP_OK
      else:
        substitutionCost = costs[i-1][j-1] + 1 # penalty is always 1
        insertionCost = costs[i][j-1]      + 1 # penalty is always 1
        deletionCost= costs[i-1][j]        + 1 # penalty is always 1

        costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
        if costs[i][j] == substitutionCost:
          backtrace[i][j] = OP_SUB
        elif costs[i][j] == insertionCost:
          backtrace[i][j] = OP_INS
        else:
          backtrace[i][j] = OP_DEL

  # back trace though the best route:
  i = len(r)
  j = len(h)
  numSub = 0
  numDel = 0
  numIns = 0
  numCor = 0

  #new df to track where an error was made
  tracker_df = pd.DataFrame({'phoneme':r,'error':['NIL' for i in range(len(r))],'substituted':['NIL' for i in range(len(r))]})

  while i > 0 or j > 0:
    if backtrace[i][j] == OP_OK:
      numCor += 1
      i-=1
      j-=1
    elif backtrace[i][j] == OP_SUB:
      # edit df
      tracker_df.at[i-1,"error"] = "Substitution"
      tracker_df.at[i-1,'substituted'] = h[j-1]
      # print(f"There was a substitution of {h[j-1]} for {r[i-1]} between {r[i-2]} and {r[i]}.")
      numSub +=1
      i-=1
      j-=1
    elif backtrace[i][j] == OP_INS:
      numIns += 1
      # try:
      #   print(f"There was an insertion of {h[j-1]} between {r[i-1]} and {r[i]}.")
      # except:
      #   print(f"There was an insertion of {h[j-1]} after {r[i-1]}")
      j-=1
    elif backtrace[i][j] == OP_DEL:
      numDel += 1
      # edit df
      tracker_df.at[i-1,"error"] = "Deletion"
      # try:
      #   print(f"There was a deletion     of {r[i-1]} between {r[i-2]} and {r[i]}.")
      # except:
      #   print(f"There was a deletion     of {r[i-1]} after {r[i-2]}")
      i-=1
  per_result = round( (numSub + numDel + numIns) / (float) (len(r)), 3)
  return per_result


def evaluate_per(model, tokenizer, test_loader):
    model.eval()
    total_per = 0
    total_sequences = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)

            predicted_ids = model.generate(input_ids)
            predicted_phonemes = [tokenizer.decode(ids, skip_special_tokens=True).split() for ids in predicted_ids]
            true_phonemes = [tokenizer.decode(ids, skip_special_tokens=True).split() for ids in batch['labels']]

            # Рассчет PER для каждой последовательности
            for true_seq, pred_seq in zip(true_phonemes, predicted_phonemes):
                per = error_rate(" ".join(true_seq), " ".join(pred_seq))
                print()
                if per >= 1:
                  per += 1
                total_per += per
                total_sequences += 1

    return total_per / total_sequences
