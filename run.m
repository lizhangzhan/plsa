% a runnable demo to show plsa in nlp application
global MAXNUMDIM;
MAXNUMDIM = 20000; % global variable, dimension of terms
global MAXNUMDOC;
MAXNUMDOC = 2000;  % global variable, number of documents
numTopic = 16;     % number of topics
numIter = 50;      % number of iteration

% 1th, preprocess the raw text set
[term2Index, index2Term, termDocMatrix] = analyze('data/award_abstracts.txt');

fprintf('Num of dimension: %d\n', size(termDocMatrix, 1));
fprintf('Num of document: %d\n', size(termDocMatrix, 2));

% 2th, fit a plsa model from a given term-doc matrix
[prob_term_topic, prob_topic_doc, lls] = plsa(termDocMatrix, numTopic, numIter);

% 3th, display topN keywords for each topic
topN = 50;
for z = 1:numTopic
	fprintf('TopN(%d) keywords for topic %d\n', topN, z);
	showTopN(index2Term, prob_term_topic(:, z), topN);
end

% 4th, plot the log-likelihood per iteration
figure;
plot(lls);
xlabel('Iteration');
ylabel('log-likehood');
