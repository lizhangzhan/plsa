function [prob_term_topic, prob_doc_topic, prob_topic, lls] = plsa2(termDocMatrix, numTopic, iter)
% Fit a plsa model from a given term-document matrix

[numTerm, numDoc] = size(termDocMatrix);

prob_topic = rand(numTopic, 1); % p(topic)
prob_topic(:) = sum(prob_topic(:)); % normalization

prob_term_topic = rand(numTerm, numTopic); % p(term | topic)
for z = 1:numTopic
	prob_term_topic(:, z) = prob_term_topic(:, z) / sum(prob_term_topic(:, z)); % normalization
end

prob_doc_topic = rand(numDoc, numTopic);   % p(doc | topic)
for z = 1:numTopic
	prob_doc_topic(:, z) = prob_doc_topic(:, z) / sum(prob_doc_topic(:, z)); % normalization
end

prob_term_doc = rand(numTerm, numDoc); % sum_{topic}{p(topic | doc, term)}

prob_topic_term_doc = cell(numTopic, 1);   % p(topic | doc, term)

for z = 1 : numTopic
	prob_topic_term_doc{z} = zeros(numTerm, numDoc);
end

lls = []; % maximum log-likelihood estimations

for i = 1 : iter
	disp('E-step...');
	for d = 1:numDoc
		%fprintf('processing doc %d\n', d);
		w = find(termDocMatrix(:, d));
		prob_term_doc(w, d) = 0;
		for z = 1:numTopic
			prob_topic_term_doc{z}(w, d) = prob_topic(z) .* prob_doc_topic(d, z) .* prob_term_topic(w, z);
			prob_term_doc(w, d) = prob_term_doc(w, d) + prob_topic_term_doc{z}(w, d);
		end
		for z = 1:numTopic
			prob_topic_term_doc{z}(w, d) = prob_topic_term_doc{z}(w, d) ./ prob_term_doc(w, d); % normalization
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
		prob_doc_topic(:, z) = prob_doc_topic(:, z) / prob_topic(z); % normalization
		assert((sum(prob_doc_topic(:, z)) - 1.0) < 1e-6)
	end
	disp('Update p(word | topic)...');
	for z = 1:numTopic
		for w = 1:numTerm
			d = find(termDocMatrix(w, :));
			prob_word_topic(w, z) = sum(termDocMatrix(w, d) .* prob_topic_term_doc{z}(w, d));
		end
		assert(prob_topic(z) - sum(prob_word_topic(:, z)) < 1e-6);
		prob_word_topic(:, z) = prob_word_topic(:,z) / prob_topic(z); % normalization
		assert((sum(prob_word_topic(:, z)) - 1.0) < 1e-6)
	end
	
	disp('Update p(topic)...');
	prob_topic(:) = prob_topic(:) / sum(prob_topic(:)); % normalization
	assert((sum(prob_topic(:)) - 1.0) < 1e-6);

	% calculate likelihood and update p(term, doc)
	fprintf('Iteration %d\n', i);
	disp('Calculate maximum likelihood...');
	ll = 0;
	for d = 1:numDoc
		prob_term_doc(:, d) = 0;
		w = find(termDocMatrix(:, d));
		for z = 1:numTopic
			prob_term_doc(w, d) = prob_term_doc(w, d) + prob_topic(z) .* prob_doc_topic(d, z) .* prob_term_topic(w, z);
		end
		ll = ll + sum(termDocMatrix(w, d) .* log(prob_term_doc(w, d)));
	end
	fprintf('likelihood: %f\n', ll);
	lls= [lls;ll];
end
save model.mat prob_doc_topic prob_word_topic prob_topic
end
