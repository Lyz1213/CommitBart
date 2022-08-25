# Downstream generation tasks
We provide the code for reproducing the experiments on Positive Code Statements Generation.
## Data preprocess
Given a commit is made up of several components. We defined several special sep token to denote each ->:

    *''[MSG]'' for commit message
    *''[FILE]'' for file path
    *''[CODE]'' For code snippet
    *''[POS]/[END]'' for a postive change statement
    *''[NEG]/[END]'' for a negative change statement

Besides, we also use segment embedding to embed tokens in each segment in CommitBART-base -> 

    *0: commit message
    *1: postive statmens
    *2: negative statements
    *3: file path
    *4: code context