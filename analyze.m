function [word2Index, index2Word, termDocMatrix] = analyze(input)
% load documents from a disk file, pre-process
global MAXNUMDIM;
global MAXNUMDOC;
word2Index = containers.Map;
index2Word = cell(MAXNUMDIM, 1);
termDocMatrix = sparse(MAXNUMDIM, MAXNUMDOC);
numDoc = 0;
numWord = 0;
index = 0;

% Prepare read the input file

disp('Open raw data file...');
fp = fopen(input);
while ~feof(fp)
	numDoc = numDoc + 1;
	line = regexprep(lower(fgetl(fp)), '[^\w]', ' ');
	terms = textscan(line, '%s');
	for i = 1:length(terms{1});
		term = terms{1}{i};
		if length(term) < 4 || instopwords(term), continue; end
		term = porterStemmer(term);
		if ~isKey(word2Index, term)
			index = numWord + 1;
			word2Index(term) = index;
			index2Word{index} = term;
			numWord = numWord + 1;
		else
			index = word2Index(term);
		end
		termDocMatrix(index, numDoc) = termDocMatrix(index, numDoc) + 1;
	end
end
fclose(fp);

termDocMatrix = termDocMatrix(1:numWord, 1:numDoc);
index2Word = index2Word(1:numWord);

% remove the common word, such stop words, a, an ...
for w = 1:numWord
	if length(find(termDocMatrix(w, :))) > numDoc * 0.5
		termDocMatrix(w, :) = 0;
	end
end

function rev = instopwords(word)
persistent stopwords; 
stopwords = {'that', 'this', 'these','those','were'};
rev = 0;
for i = 1:length(stopwords)
	if strcmp(stopwords{i}, word) == 1
		rev = 1;
	end
end
