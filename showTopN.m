function showTopN(index2Term, prob_term_topic, topN)
	[S, I] = sort(prob_term_topic, 'descend');
	for w = I(1:topN)'
		fprintf('%s\t(%f)\n', index2Term{w}, prob_term_topic(w));
	end
end
