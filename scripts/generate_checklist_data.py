# to setup beforehand
# git clone https://github.com/marcotcr/checklist
# cd checklist
# pip install -e .
# tar xvzf release_data.tar.gz
# cd ..

import json
with open('data/LittleGLUE/SST-2/checklist.jsonl', 'w') as outf:
	with open('checklist/release_data/sentiment/tests_n500', 'r') as f:
		for line in f:
                        # we hardcode the label to 0 just so the code doesnt fail downstream (we ignore the accuracy)
			mydict = {'sentence': line.strip(), 'label': "0", 'idx': None}
			json.dump(mydict, outf)
			outf.write('\n')

with open('data/LittleGLUE/QQP/checklist.jsonl', 'w') as outf:
        with open('checklist/release_data/qqp/tests_n500', 'r') as f:
            for i, line in enumerate(f):
                    if i == 0:
                        continue
                    # we hardcode the label to 0 just so the code doesnt fail downstream (we ignore the accuracy)
                    mydict = {'question1': line.split('\t')[1].strip(), 'question2': line.split('\t')[2].strip(), 'label': "0", 'qidx1': 0, 'qidx2': 1}
                    json.dump(mydict, outf)
                    outf.write('\n')
