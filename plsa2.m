function [prob_term_topic, prob_topic_doc, lls] = plsa2(termDocMatrix, numTopic, iter)
% Fit a plsa model from a given term-document matrix

[numTerm, numDoc] = size(termDocMatrix);

prob_topic = rand(numTopic, 1); % p(topic)
prob_topic(:) = sum(prob_topic(:)); % normalization

prob_term_topic = rand(numTerm, numTopic); % p(term | topic)
for i = 1:numTopic
	prob_term_topic(:, i) = prob_term_topic(:, i) / sum(prob_term_topic(:, i)); % normalization
end

prob_doc_topic = rand(numDoc, numTopic);   % p(doc | topic)
for i = 1:numTopic
	prob_doc_topic(:, i) = prob_doc_topic(:, i) / sum(prob_doc_topic(:, i)); % normalization
end

prob_topic_term_doc = cell(numTopic, 1);   % p(topic | doc, term)
prob_term_doc = zeros(numTerm, numDoc);

for z = 1 : numTopic
	prob_topic_term_doc{z} = zeros(numTerm, numDoc);
end

prob_term_doc = zeros(numTerm, numDoc);

lls = []; % maximum log-likelihood estimations

for i = 1 : iter
	disp('E-step...');
	for d = 1:numDoc
		%fprintf('processing doc %d\n', d);
		prob_term_doc(:, d) = 0;
		for z = 1:numTopic
			prob_topic_term_doc{z}(:, d) = prob_topic(z) .* prob_doc_topic(d, z) .* prob_term_topic(:, z);
			prob_term_doc(:, d) = prob_term_doc(:, d) + prob_topic_term_doc{z}(:, d);
		end
		for z = 1:numTopic
			prob_topic_term_doc{z}(:, d) = prob_topic_term_doc{z}(:, d) ./ prob_term_doc(:, d);
		end
	end
	
	disp('M-step..');
	disp('Update p(doc | topic)...');
	for z = 1:numTopic
		for d = 1:numDoc
			w = find(termDocMatrix(:, d));
			prob_doc_topic(d, z) = sum(termDocMatrix(w, d) .* prob_topic_term_doc{z}(w, d));
		end
		prob_topic(z) = sum(prob_doc_topic(:, z));
		prob_doc_topic(:, z) = prob_doc_topic(:, z) / prob_topic(z);
	end
	disp('Update p(word | topic)...');
	for z = 1:numTopic
		for w = 1:numTerm
			d = find(termDocMatrix(w, :));
			prob_word_topic(w, z) = sum(termDocMatrix(w, d) .* prob_topic_term_doc{z}(w, d));
		end
		assert(prob_topic(z) - sum(prob_word_topic(:, z)) < 1e-5);
		% formatSpec = 'topic %d prob: %f \t %f';
		% fprintf(formatSpec, prob_topic(z), sum(prob_word_topic(:, z)));
		prob_word_topic(:, z) = prob_word_topic(:,z) / sum(prob_word_topic(:,z));
	end
	
	disp('Update p(topic)...');
	prob_topic(:) = prob_topic(:) / sum(prob_topic(:));

	% calculate likelihood
	fprintf('Iteration %d\n', i);
	disp('Calculate maximum likelihood...');
	ll = 0;
	for d = 1: numDoc
		for w = find(termDocMatrix(:, d))
			ll = ll + sum(termDocMatrix(w, d) .* log(prob_term_doc(w,d)));
		end
	end
	fprintf('likelihood: %f\n', ll);
	lls= [lls;ll];
end
save model.mat prob_doc_topic prob_word_topic prob_topic
end
