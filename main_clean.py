import copy
import os
import time
import jsonlines
import openai
import json
import re
import numpy as np
from utils import delete_extra_zero,_strip_string
import argparse
from statistics import mean
from collections import Counter
import traceback

# OpenAI Key
openai.api_key = "Put Your Key Here"









def find_math_answer(s):

    assert('boxed' in s)
    # s = s.replace(",", "")
    ans = s.split('boxed')[-1]
    if(ans[0] == '{'):
        stack = 1
        a = ''
        for c in ans[1:]:
            if(c == '{'):
                stack += 1
                a += c
            elif(c == '}'):
                stack -= 1
                if(stack == 0): break
                a += c
            else:
                a += c
    else:
        a = ans.split('$')[0].strip()
    a=_strip_string(a)
    return a

def extract_math_answer(pred_str):
    if('The answer is ' in pred_str):
        pred = pred_str.split('The answer is ')[-1].strip()
    elif('the answer is ' in pred_str):
        pred = pred_str.split('the answer is ')[-1].strip()
    elif 'boxed' in pred_str:
        ans = pred_str.split('boxed')[-1]
        if (ans[0] == '{'):
            stack = 1
            a = ''
            for c in ans[1:]:
                if (c == '{'):
                    stack += 1
                    a += c
                elif (c == '}'):
                    stack -= 1
                    if (stack == 0): break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        a = _strip_string(a)
        pred=a

    else:
        pattern = '-?\d*\.?\d+'
        pred = re.findall(pattern, pred_str)
        if(len(pred) >= 1):
            # print(pred_str)
            pred = pred[-1]
        else: pred = ''
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
        if pred[-1] == "/":
            pred = pred[:-1]
    pred=_strip_string(pred)
    if 'boxed' in pred:
        ans = pred.split('boxed')[-1]
        if (ans[0] == '{'):
            stack = 1
            a = ''
            for c in ans[1:]:
                if (c == '{'):
                    stack += 1
                    a += c
                elif (c == '}'):
                    stack -= 1
                    if (stack == 0): break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        a = _strip_string(a)
        pred=a
    return pred

def data_reader(args):
    questions = []
    answers = []
    decoder = json.JSONDecoder()

    if args.dataset == "aqua":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "(" + "(".join(json_res["options"])
                choice = choice.replace("(", " (").replace(")", ") ")
                choice = "Answer Choices:" + choice
                questions.append(json_res["question"].strip() + " " + choice)
                answers.append(json_res["correct"])
    elif args.dataset in ["algebra","counting_and_probability","geometry","intermediate_algebra","number_theory","prealgebra","precalculus"]:

        for filename in os.listdir(args.dataset_path):
            if (filename.endswith('.json')):
                d = json.load(open(args.dataset_path + '/' + filename))
                questions.append(d['problem'])
                answers.append(find_math_answer(d['solution']))
    elif args.dataset == "gsm8k":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                questions.append(json_res["question"].strip())
                answers.append(delete_extra_zero(json_res["answer"].split("#### ")[-1].replace(",", "")))

    elif args.dataset in ("addsub", "multiarith", "singleeq"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["sQuestion"].strip()
                a = str(line["lSolutions"][0])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(delete_extra_zero(a))


    elif args.dataset == "svamp":
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["Body"].strip() + " " + line["Question"].strip()
                a = str(line["Answer"])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(delete_extra_zero(a))

    else:
        raise ValueError("dataset is not properly defined ...")

    q_len_list = []
    for q in questions:

        q_len_list.append(len(q.split(" ")))
    q_len_mean = mean(q_len_list)

    print("dataset : {}".format(args.dataset))
    print("data size : {}".format(len(answers)))
    print("average num of words for each sample : {}".format(q_len_mean))

    return questions, answers


def answer_cleansing(args, pred):
    # print("pred_before : " + pred)

    if args.dataset in ["algebra","counting_and_probability","geometry","intermediate_algebra","number_theory","prealgebra","precalculus"]:
        # pred = pred.replace(",", "")
        pred_final=extract_math_answer(pred)
        return pred_final

    if args.method in ("few_shot", "few_shot_cot"):
        preds = pred.split(args.direct_answer_trigger_for_fewshot)
        answer_flag = True if len(preds) > 1 else False
        pred = preds[-1]

    if args.dataset in ("aqua"):
        pred = pred.upper()
        # print(pred)
        pred = re.findall(r'A|B|C|D|E', pred)


    elif args.dataset in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq"):
        pred=pred.replace(",", "")
        pred = [delete_extra_zero(s.replace(",", "")) for s in re.findall(r'-?\d+/?\.?\d*', pred)]
    else:
        raise ValueError("dataset is not properly defined ...")

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if args.method in ("few_shot", "few_shot_cot"):
            if answer_flag:
                # choose the first element in list ...
                pred = pred[0]
            else:
                # choose the last element in list ...
                pred = pred[-1]
        elif args.method in ("zero_shot", "zero_shot_cot"):
            # choose the first element in list ...
            pred = pred[0]
        else:
            raise ValueError("method is not properly defined ...")

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
        if pred[-1] == "/":
            pred = pred[:-1]


    return pred





def answer_clean(args, pred):
    if args.dataset in ['aqua']:
        pred_answer = answer_cleansing(args, copy.deepcopy(pred))

        return pred_answer, "({})".format(pred_answer)
    else:
        pred_answer = answer_cleansing(args, copy.deepcopy(pred))
        return pred_answer, pred_answer


def answer_clean_all(args, pred):
    pred_answer_list = []
    pred_answer_list_hint_list = []


    for index_len in range(len(pred)):
        # print(pred[index_len]['text'])
        if args.eng in ["gpt-3.5-turbo", "gpt-4"]:
            pred_answer, pred_answer_hint = answer_clean(args, pred[index_len]["message"]["content"].strip())
        else:
            pred_answer, pred_answer_hint = answer_clean(args, pred[index_len]["text"].strip())

        pred_answer_list.append(pred_answer)
        pred_answer_list_hint_list.append(pred_answer_hint)
    most_answer=Counter(copy.deepcopy(pred_answer_list)).most_common()[0][0]
    pred_answer_list_hint_list_new=[]
    for element in pred_answer_list_hint_list:
        if element.__contains__(most_answer):
            pred_answer_list_hint_list_new.append(element)
    pred_answer_list_hint_list=pred_answer_list_hint_list_new
    return Counter(copy.deepcopy(pred_answer_list)).most_common()[0][0], Counter(pred_answer_list_hint_list).most_common()[0][0]



def create_response(prompt_input, eng='text-davinci-002', max_tokens=256, temperature=0.0, stop="Q"):
    response = openai.Completion.create(
        engine=eng,
        prompt=prompt_input,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["{}:".format(stop)]
    )
    return response

def create_response_chat(prompt_input, eng='text-davinci-002', max_tokens=256, temperature=0.0, stop="Q"):
    response = openai.ChatCompletion.create(
        model=eng,
        messages=prompt_input,
        temperature=temperature,
    )
    return response


def create_repeat_list(input_string, number):
    prompt_input_list = []
    for index_sample in range(number):
        prompt_input_list.append(input_string)

    return prompt_input_list


def get_answer_from_gpt_sample(prompt, question, eng='text-davinci-002', max_tokens=256, temperature=0.0,
                               question_string="ori", args=None):
    # batch=20
    count_total = args.sample
    response_all = []
    if eng in  ["gpt-3.5-turbo","gpt-4"]:
        if question_string == "ori":
            prompt_input = prompt + "\n\nQ: {}\nA:".format(question)

            response = create_response_chat([
                        {"role": "system", "content": "Follow the given examples and answer the question."},
                        {"role": "user", "content": prompt_input},
                    ], eng, max_tokens, temperature, "Q")
            response_all = response['choices']


        elif question_string == "complex":
            prompt_input = prompt + "\n\nQuestion: {}\nA:".format(question)
            response = create_response_chat([
                        {"role": "system", "content": "Follow the given examples and answer the question."},
                        {"role": "user", "content": prompt_input},
                    ], eng, max_tokens, temperature, "Question")
            response_all = response['choices']
        return response['choices'][0]['message']["content"].strip(), response_all







    else:
        if question_string == "ori":
            prompt_input = prompt + "\n\nQ: {}\nA:".format(question)
            if args.sample > 1:
                # prompt_input_list=[]
                while count_total > 0:
                    repeat_this = min(5, count_total)
                    prompt_input_list = create_repeat_list(prompt_input, repeat_this)
                    response = create_response(prompt_input_list, eng, max_tokens, temperature)
                    response_all.extend(response['choices'])
                    count_total = count_total - repeat_this

            else:
                response = create_response(prompt_input, eng, max_tokens, temperature, "Q")
                response_all = response['choices']


        elif question_string == "complex":
            prompt_input = prompt + "\n\nQuestion: {}\nA:".format(question)
            # print(prompt_input)
            if args.sample > 1:
                # prompt_input_list=[]
                while count_total > 0:
                    repeat_this = min(5, count_total)
                    prompt_input_list = (create_repeat_list(prompt_input, repeat_this))
                    response = create_response(prompt_input_list, eng, max_tokens, temperature)
                    response_all.extend(response['choices'])
                    count_total = count_total - repeat_this
            else:
                response = create_response(prompt_input, eng, max_tokens, temperature, "Question")
                response_all = response['choices']
    # print(response)
        return response['choices'][0]['text'].strip(), response_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_dir", default=None, type=str, required=True, help="directory to prompt file (.txt)")
    parser.add_argument("--hint", default="None", type=str,
                        help="directory to progressive-hint prompt file (.txt)")
    parser.add_argument("--eng", default=None, type=str, required=True, help="engine")
    parser.add_argument("--num_test", default=400, type=int, help="number of samples tested. -1 if on all test samples")
    parser.add_argument("--seed", default=1357, type=int, help="random seed")
    parser.add_argument("--temp", default=0.0, type=float, help="temperature for generation")
    parser.add_argument("--temp2", default=-1, type=float, help="temperature for progressive-hint generation")
    parser.add_argument("--max_tokens", default=1024, type=int, help="max # of tokens for generation")
    parser.add_argument("--test_ind", default=None, type=str,
                        help="dir to test indices. If not provided, randomly choose.")
    parser.add_argument("--suffix", default="", type=str, help="")

    parser.add_argument("--hint_length", default=2, type=int,
                        help="return after the last hint_lenght answers are the same")
    parser.add_argument("--direct_answer_trigger_for_fewshot", default="The answer is", type=str,
                        help="used for extract answer")
    parser.add_argument("--method", default="few_shot", type=str,
                        help="we use prompt so that the method is few-shot")
    parser.add_argument("--dataset", default="", type=str,
                        help="the dataset name")
    parser.add_argument("--q1", default="ori", type=str,choices=["ori","complex"],
                        help="q1 is used for base prompt. ori for standard prompt and CoT; compelx for Complex CoT")
    parser.add_argument("--q2", default="ori", type=str,choices=["ori","complex"],
                        help="q2 is used for progressive-hint prompt. ori for standard prompt and CoT; compelx for Complex CoT")
    parser.add_argument("--sample", default=1, type=int,
                        help="sample path number")
    parser.add_argument("--sleep", default=1, type=int,
                        help="sleep time after error.")

    args = parser.parse_args()

    def make_print_to_file(file_name='./', path="./"):
        '''
        path， it is a path for save your log about fuction print
        example:
        use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
        :return:
        '''
        import os
        import sys
        import time
        import datetime
        import pytz

        class Logger(object):
            def __init__(self, filename, path="./"):
                self.terminal = sys.stdout
                self.log = open(filename, "w", encoding='utf8')

            def write(self, message):
                # self.terminal.write(message)
                if message != "\n" and message != "\r":
                    message = str(datetime.datetime.now(pytz.timezone('Asia/Shanghai'))) + "   " + message

                self.terminal.write(message)
                self.log.write(message)
                self.log.flush()

            def flush(self):
                pass

        # fileName = time.strftime(file_name+'%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        sys.stdout = Logger(file_name, path=path)

    print(args)
    dataset_path = {"addsub": "AddSub/AddSub.json", "aqua": "AQuA/AQuA.json", "gsm8k": "gsm8k/gsm8k.jsonl",
                    "multiarith": "MultiArith/MultiArith.json", "singleeq": "SingleEq/SingleEq.json",
                    "svamp": "SVAMP/SVAMP.json", "digit5": "digit5/digit5.pkl", "digit3": "digit3/digit3.pkl",
                    "digit4": "digit4/digit4.pkl", "digit6": "digit6/digit6.pkl", "digit7": "digit7/digit7.pkl",
                    "algebra": "math/algebra",
                    "counting_and_probability": "math/counting_and_probability", "geometry": "math/geometry",
                    "intermediate_algebra": "math/intermediate_algebra", "number_theory": "math/number_theory",
                    "prealgebra": "math/prealgebra", "precalculus": "math/precalculus", }

    args.dataset_path = "dataset/{}".format(dataset_path[args.dataset])

    # load prompts
    file = args.prompt_dir
    assert file.endswith(".txt")
    prompt_name = os.path.basename(file)[:-4]
    print(file, prompt_name)
    with open(file, "r", encoding='utf-8') as f:
        prompt = f.read().strip()

    if args.hint != "None":
        with open(args.hint, "r", encoding='utf-8') as f:
            prompt_hint = f.read().strip()

    questions, answers = data_reader(args)
    qa_pairs = [(questions[idx], answers[idx]) for idx in range(len(questions))]
    print("loading dataset complete. altogether", len(qa_pairs), "questions")

    if args.temp2 < 0:
        args.temp2 = args.temp

    # scale down. -1 if not.
    NUM_TEST = args.num_test
    if NUM_TEST == -1:
        qa_pairs_test = qa_pairs
    else:
        if args.test_ind is None:
            np.random.seed(args.seed)
            rand_indices = np.random.choice(len(qa_pairs), NUM_TEST, replace=False)
            qa_pairs_test = [qa_pairs[i] for i in rand_indices]
        else:
            with open(args.test_ind, "r") as f:
                test_ind = json.load(f)
            assert len(test_ind) == NUM_TEST
            qa_pairs_test = [qa_pairs[i] for i in test_ind]

    print("testing on", len(qa_pairs_test), "samples")




    #Store PHP Answer
    file_name = "results/{}/Sample-{}_T-{}_T-{}/Hint-{}{}-{}.eng{}.sample{}.seed{}.temp{}.{}.jsonl".format(args.dataset,
                                                                                                      args.sample,
                                                                                                      args.temp,
                                                                                                      args.temp2,
                                                                                                      os.path.basename(
                                                                                                          args.hint)[
                                                                                                      :-4],
                                                                                                      args.hint_length,
                                                                                                      prompt_name,
                                                                                                      args.eng,
                                                                                                      NUM_TEST,
                                                                                                      args.seed,
                                                                                                      args.temp,
                                                                                                      args.suffix)
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    writer = jsonlines.open(file_name, mode='w')

    # Store non-PHP Answer
    file_name_none = "results/{}/Sample-{}_T-{}_T-{}/None{}-{}.eng{}.sample{}.seed{}.temp{}.{}.jsonl".format(
        args.dataset, args.sample, args.temp, args.temp2, args.hint_length,
        prompt_name, args.eng, NUM_TEST, args.seed, args.temp, args.suffix)
    writer_none = jsonlines.open(file_name_none, mode='w')

    #make log file
    make_print_to_file(
        "results/{}/Sample-{}_T-{}_T-{}/{}{}-{}.eng{}.sample{}.seed{}.temp{}.{}.txt".format(args.dataset, args.sample,
                                                                                            args.temp, args.temp2,
                                                                                            os.path.basename(args.hint)[
                                                                                            :-4], args.hint_length,
                                                                                            prompt_name, args.eng,
                                                                                            NUM_TEST, args.seed,
                                                                                            args.temp, args.suffix))

    #Print Log File Name
    print("results/{}/Sample-{}_T-{}_T-{}/{}{}-{}.eng{}.sample{}.seed{}.temp{}.{}.txt".format(args.dataset, args.sample,
                                                                                              args.temp, args.temp2,
                                                                                              os.path.basename(
                                                                                                  args.hint)[:-4],
                                                                                              args.hint_length,
                                                                                              prompt_name, args.eng,
                                                                                              NUM_TEST, args.seed,
                                                                                              args.temp, args.suffix))

    #print Dataset length
    print("loading dataset complete. altogether", len(qa_pairs), "questions")
    print("testing on", len(qa_pairs_test), "samples")
    print(args)

    count = 0
    correct = 0.0
    correct_none = 0.0
    statics_interact = []
    for (question, answer) in qa_pairs_test:
        count += 1 # how many QA until now
        print()
        print()
        print("currently", prompt_name, "#", count)
        result = dict()
        result['question'] = question
        result['answer'] = answer

        result = dict()
        result['question'] = question
        result['answer'] = answer

        result_none = dict()
        result_none['question'] = question
        result_none['answer'] = answer
        max_tokens = args.max_tokens

        count_this = 0
        answer_previous_number_array = []
        answer_previous_number_array2=[]

        while True:
            # time.sleep(1)
            print("***Step: {}".format(count_this))
            print("***{} (Hint: The answer is near to {}).".format(question, ', '.join(
                ('%s' % id for id in answer_previous_number_array))))
            if count_this == 0: #for the base interaction
                try:
                    #try to get answer
                    answer_first, answer_this = get_answer_from_gpt_sample(prompt, question,
                                                                           eng=args.eng, max_tokens=max_tokens,
                                                                           temperature=args.temp,
                                                                           question_string=args.q1, args=args)

                    result_none['ans_' + prompt_name] = answer_first
                    print("***Len {}".format(len(answer_this)))
                    print("***" + answer_first)

                    # clean answer.
                    # answer_this_number_check: check answer is correct or not.
                    #answer_this_hint: used for answer hint
                    answer_this_number_check, answer_this_hint = answer_clean_all(args, answer_this)

                    answer_previous_number = answer_this_hint
                    answer_previous_number_array.append(answer_this_number_check)
                    answer_previous_number_array2.append(answer_previous_number)#used for hint
                    count_this = count_this + 1#interaction number update
                    if answer_this_number_check == answer:
                        correct_none += 1
                    result_none['ans_correct_ratio'] = correct_none / count
                    print("***Answer:" + str(answer_this_number_check) + " " +str(answer_this_hint) + " " + str(answer) + " " + str(
                        correct_none / count))
                except Exception as e:
                    print(repr(e))
                    traceback.print_exc()
                    time.sleep(args.sleep)


            else:
                # print(question)

                try:
                    question_new = "{} (Hint: The answer is near to {}).".format(question, ', '.join(
                        ('%s' % id for id in answer_previous_number_array2)))
                    print("***{}".format(question_new))
                    answer_first, answer_this = get_answer_from_gpt_sample(prompt_hint, question_new,
                                                                           eng=args.eng, max_tokens=max_tokens,
                                                                           temperature=args.temp2,
                                                                           question_string=args.q2, args=args)
                    print("***Len {}".format(len(answer_this)))
                    print("***" + answer_first)

                    # clean answer.
                    # answer_this_number_check: check answer is correct or not.
                    #answer_this_hint: used for answer hint
                    answer_this_number_check, answer_this_hint = answer_clean_all(args, answer_this)
                    print("***Answer:" + str(answer_this_number_check) + " " +str(answer_this_hint) + " " + str(answer))

                    count_this += 1
                    answer_previous_number_array.append(answer_this_number_check)
                    answer_previous_number_array2.append(answer_this_hint)

                    if len(list(set(answer_previous_number_array[-min(len(answer_previous_number_array),
                                                                      args.hint_length):]))) == 1 and count_this>=args.hint_length:
                        break
                    answer_previous_number = answer_this_hint
                except Exception as e:
                    print(repr(e))
                    traceback.print_exc()
                    time.sleep(args.sleep)

        result['ans_' + prompt_name] = answer_this
        if answer_this_number_check == answer:
            correct += 1
        result['ans_correct_ratio'] = correct / count
        print("***Answer:" + str(answer_this_number_check) + " " +str(answer_this_hint) + " " + str(answer) + " " + str(correct / count))


        writer.write(result)
        writer_none.write(result_none)
        statics_interact.append(count_this)

    # the last element is the prompt
    writer.write(prompt)
    writer.write(prompt_hint)
    writer.close()

    writer_none.write(prompt)
    writer_none.close()
    statics_interact = np.array(statics_interact)
    print("Interact: Mean {} & Std {}".format(statics_interact.mean(), statics_interact.std()))


if __name__ == '__main__':
    main()
