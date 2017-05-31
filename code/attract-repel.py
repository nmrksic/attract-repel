import ConfigParser
import numpy
import sys
import time
import random 
import math
import os
from copy import deepcopy
import json
from numpy.linalg import norm
from numpy import dot
import codecs
from scipy.stats import spearmanr
import tensorflow as tf

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


class ExperimentRun:
    """
    This class stores all of the data and hyperparameters required for an Attract-Repel run. 
    """

    def __init__(self, config_filepath):
        """
        To initialise the class, we need to supply the config file, which contains the location of
        the pretrained (distributional) word vectors, the location of (potentially more than one)
        collections of linguistic constraints (one pair per line), as well as the  
        hyperparameters of the Attract-Repel procedure (as detailed in the TACL paper).
        """
        self.config = ConfigParser.RawConfigParser()
        try:
            self.config.read(config_filepath)
        except:
            print "Couldn't read config file from", config_filepath
            return None

        distributional_vectors_filepath = self.config.get("data", "distributional_vectors")

        try:
            self.output_filepath = self.config.get("data", "output_filepath")
        except:
            self.output_filepath = "results/final_vectors.txt"

        # load initial distributional word vectors. 
        distributional_vectors = load_word_vectors(distributional_vectors_filepath)
        
        if not distributional_vectors:
            return

        print "SimLex score (Spearman's rho coefficient) of initial vectors is:\n" 
        simlex_scores(distributional_vectors)

        self.vocabulary = set(distributional_vectors.keys())

        # this will be used to load constraints 
        self.vocab_index = {}
        self.inverted_index = {}

        for idx, word in enumerate(self.vocabulary):
            self.vocab_index[word] = idx
            self.inverted_index[idx] = word

        # load list of filenames for synonyms and antonyms. 
        synonym_list = self.config.get("data", "synonyms").replace("[","").replace("]", "").replace(" ", "").split(",")
        antonym_list = self.config.get("data", "antonyms").replace("[","").replace("]", "").replace(" ", "").split(",")

        self.synonyms = set()
        self.antonyms = set()

        if synonym_list != "":
            # and we then have all the information to load all linguistic constraints
            for syn_filepath in synonym_list:
                if syn_filepath != "":
                    self.synonyms = self.synonyms | self.load_constraints(syn_filepath)
        else:
            self.synonyms = set()

        if antonym_list != "":
            for ant_filepath in antonym_list:
                if ant_filepath != "":
                    self.antonyms = self.antonyms | self.load_constraints(ant_filepath)
        else:
            self.antonyms = set()

        # finally, load the experiment hyperparameters:
        self.load_experiment_hyperparameters()

        self.embedding_size = random.choice(distributional_vectors.values()).shape[0]
        self.vocabulary_size = len(self.vocabulary)

        # Next, prepare the matrix of initial vectors and initialise the model. 

        numpy_embedding = numpy.zeros((self.vocabulary_size, self.embedding_size), dtype="float32")
        for idx in range(0, self.vocabulary_size):
            numpy_embedding[idx, :] = distributional_vectors[self.inverted_index[idx]]

        # load the handles so that we can load current state of vectors from the Tensorflow embedding. 
        embedding_handles = self.initialise_model(numpy_embedding)
        
        self.embedding_attract_left = embedding_handles[0]
        self.embedding_attract_right = embedding_handles[1]
        self.embedding_repel_left = embedding_handles[2]
        self.embedding_repel_right = embedding_handles[3]

        init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init)


    def initialise_model(self, numpy_embedding):
        """
        Initialises the TensorFlow Attract-Repel model.
        """
        self.attract_examples = tf.placeholder(tf.int32, [None, 2]) # each element is the position of word vector. 
        self.repel_examples = tf.placeholder(tf.int32, [None, 2]) # each element is again the position of word vector.

        self.negative_examples_attract = tf.placeholder(tf.int32, [None, 2])
        self.negative_examples_repel = tf.placeholder(tf.int32, [None, 2])

        self.attract_margin = tf.placeholder("float")
        self.repel_margin = tf.placeholder("float")
        self.regularisation_constant = tf.placeholder("float")
        
        # Initial (distributional) vectors. Needed for L2 regularisation.         
        self.W_init = tf.constant(numpy_embedding, name="W_init")

        # Variable storing the updated word vectors. 
        self.W_dynamic = tf.Variable(numpy_embedding, name="W_dynamic")


        # Attract Cost Function: 

        # placeholders for example pairs...
        attract_examples_left = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.W_dynamic, self.attract_examples[:, 0]), 1) 
        attract_examples_right = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.W_dynamic, self.attract_examples[:, 1]), 1)

        # and their respective negative examples:
        negative_examples_attract_left = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.W_dynamic, self.negative_examples_attract[:, 0]), 1)
        negative_examples_attract_right = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.W_dynamic, self.negative_examples_attract[:, 1]), 1)

        # dot product between the example pairs. 
        attract_similarity_between_examples = tf.reduce_sum(tf.multiply(attract_examples_left, attract_examples_right), 1) 

        # dot product of each word in the example with its negative example. 
        attract_similarity_to_negatives_left = tf.reduce_sum(tf.multiply(attract_examples_left, negative_examples_attract_left), 1) 
        attract_similarity_to_negatives_right = tf.reduce_sum(tf.multiply(attract_examples_right, negative_examples_attract_right), 1)

        # and the final Attract Cost Function (sans regularisation):
        self.attract_cost = tf.nn.relu(self.attract_margin + attract_similarity_to_negatives_left - attract_similarity_between_examples) + \
                       tf.nn.relu(self.attract_margin + attract_similarity_to_negatives_right - attract_similarity_between_examples)

        # Repel Cost Function: 

        # placeholders for example pairs...
        repel_examples_left = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.W_dynamic, self.repel_examples[:, 0]), 1) # becomes batch_size X vector_dimension 
        repel_examples_right = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.W_dynamic, self.repel_examples[:, 1]), 1)

        # and their respective negative examples:
        negative_examples_repel_left  = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.W_dynamic, self.negative_examples_repel[:, 0]), 1)
        negative_examples_repel_right = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.W_dynamic, self.negative_examples_repel[:, 1]), 1)

        # dot product between the example pairs. 
        repel_similarity_between_examples = tf.reduce_sum(tf.multiply(repel_examples_left, repel_examples_right), 1) # becomes batch_size again, might need tf.squeeze

        # dot product of each word in the example with its negative example. 
        repel_similarity_to_negatives_left = tf.reduce_sum(tf.multiply(repel_examples_left, negative_examples_repel_left), 1)
        repel_similarity_to_negatives_right = tf.reduce_sum(tf.multiply(repel_examples_right, negative_examples_repel_right), 1)

        # and the final Repel Cost Function (sans regularisation):
        self.repel_cost = tf.nn.relu(self.repel_margin - repel_similarity_to_negatives_left + repel_similarity_between_examples) + \
                       tf.nn.relu(self.repel_margin - repel_similarity_to_negatives_right + repel_similarity_between_examples)


        # The Regularisation Cost (separate for the two terms, depending on which one is called): 

        # load the original distributional vectors for the example pairs: 
        original_attract_examples_left = tf.nn.embedding_lookup(self.W_init, self.attract_examples[:, 0])
        original_attract_examples_right = tf.nn.embedding_lookup(self.W_init, self.attract_examples[:, 1])

        original_repel_examples_left = tf.nn.embedding_lookup(self.W_init, self.repel_examples[:, 0])
        original_repel_examples_right = tf.nn.embedding_lookup(self.W_init, self.repel_examples[:, 1])

        # and then define the respective regularisation costs:
        regularisation_cost_attract = self.regularisation_constant * (tf.nn.l2_loss(original_attract_examples_left - attract_examples_left) + tf.nn.l2_loss(original_attract_examples_right - attract_examples_right))
        self.attract_cost += regularisation_cost_attract

        regularisation_cost_repel = self.regularisation_constant * (tf.nn.l2_loss(original_repel_examples_left - repel_examples_left) + tf.nn.l2_loss(original_repel_examples_right - repel_examples_right))
        self.repel_cost += regularisation_cost_repel
    
        # Finally, we define the training step functions for both steps. 

        tvars = tf.trainable_variables()
        attract_grads = [tf.clip_by_value(grad, -2., 2.) for grad in tf.gradients(self.attract_cost, tvars)]
        repel_grads = [tf.clip_by_value(grad, -2., 2.) for grad in tf.gradients(self.repel_cost, tvars)]

        attract_optimiser = tf.train.AdagradOptimizer(0.05) 
        repel_optimiser = tf.train.AdagradOptimizer(0.05) 
        
        self.attract_cost_step = attract_optimiser.apply_gradients(zip(attract_grads, tvars))
        self.repel_cost_step = repel_optimiser.apply_gradients(zip(repel_grads, tvars))

        # return the handles for loading vectors from the TensorFlow embeddings:
        return attract_examples_left, attract_examples_right, repel_examples_left, repel_examples_right


    def load_constraints(self, constraints_filepath):
        """
        This methods reads a collection of constraints from the specified file, and returns a set with
        all constraints for which both of their constituent words are in the specified vocabulary.
        """
        constraints_filepath.strip()
        constraints = set()

        with codecs.open(constraints_filepath, "r", "utf-8") as f:
            for line in f:
                word_pair = line.split()
                if word_pair[0] in self.vocabulary and word_pair[1] in self.vocabulary and word_pair[0] != word_pair[1]:
                    constraints |= {(self.vocab_index[word_pair[0]], self.vocab_index[word_pair[1]])}

        return constraints


    def load_experiment_hyperparameters(self):
        """
        This method loads/sets the hyperparameters of the procedure as specified in the paper.
        """
        self.attract_margin_value = self.config.getfloat("hyperparameters", "attract_margin")
        self.repel_margin_value = self.config.getfloat("hyperparameters", "repel_margin") 
        self.batch_size = int(self.config.getfloat("hyperparameters", "batch_size")) 
        self.regularisation_constant_value    = self.config.getfloat("hyperparameters", "l2_reg_constant")
        self.max_iter    = self.config.getfloat("hyperparameters", "max_iter")
        self.log_scores_over_time = self.config.get("experiment", "log_scores_over_time")
        self.print_simlex = self.config.get("experiment", "print_simlex")

        if self.log_scores_over_time in ["True", "true"]:
            self.log_scores_over_time = True
        else:
            self.log_scores_over_time = False

        if self.print_simlex in ["True", "true"]:
            self.print_simlex = True
        else:
            self.print_simlex = False


        print "\nExperiment hyperparameters (attract_margin, repel_margin, batch_size, l2_reg_constant, max_iter):", \
               self.attract_margin_value, self.repel_margin_value, self.batch_size, self.regularisation_constant_value, self.max_iter

    
    def extract_negative_examples(self, list_minibatch, attract_batch = True):
        """
        For each example in the minibatch, this method returns the closest vector which is not 
        in each words example pair. 
        """

        list_of_representations = []
        list_of_indices = []

        representations = self.sess.run([self.embedding_attract_left, self.embedding_attract_right], feed_dict={self.attract_examples: list_minibatch})

        for idx, (example_left, example_right) in enumerate(list_minibatch):

            list_of_representations.append(representations[0][idx])
            list_of_representations.append(representations[1][idx])

            list_of_indices.append(example_left)
            list_of_indices.append(example_right)

        condensed_distance_list = pdist(list_of_representations, 'cosine') 
        square_distance_list = squareform(condensed_distance_list)   

        if attract_batch: 
            default_value = 2.0 # value to set for given attract/repel pair, so that it can not be found as closest or furthest away. 
        else:
            default_value = 0.0 # for antonyms, we want the opposite value from the synonym one. Cosine Distance is [0,2]. 

        for i in range(len(square_distance_list)):

            square_distance_list[i,i]=default_value # NIKOLA TODO: is this wrong, does this mean that we always find the element itself as most distant?
            
            if i % 2 == 0:
                square_distance_list[i,i+1] = default_value 
            else:
                square_distance_list[i,i-1] = default_value

        if attract_batch:
            negative_example_indices = numpy.argmin(square_distance_list,axis=1) # for each of the 100 elements, finds the index which has the minimal cosine distance (i.e. most similar). 
        else:
            negative_example_indices = numpy.argmax(square_distance_list, axis=1) # for antonyms, find the least similar one. 

        negative_examples = []

        for idx in range(len(list_minibatch)):
            
            negative_example_left = list_of_indices[negative_example_indices[2 * idx]] 
            negative_example_right = list_of_indices[negative_example_indices[2 * idx + 1]]
            
            negative_examples.append((negative_example_left, negative_example_right))            

        negative_examples = mix_sampling(list_minibatch, negative_examples)

        return negative_examples


    def attract_repel(self):
        """
        This method repeatedly applies optimisation steps to fit the word vectors to the provided linguistic constraints. 
        """
        
        current_iteration = 0
        
        # Post-processing: remove synonym pairs which are deemed to be both synonyms and antonyms:
        for antonym_pair in self.antonyms:
            if antonym_pair in self.synonyms:
                self.synonyms.remove(antonym_pair)

        self.synonyms = list(self.synonyms)
        self.antonyms = list(self.antonyms)
        
        self.syn_count = len(self.synonyms)
        self.ant_count = len(self.antonyms)

        print "\nAntonym pairs:", len(self.antonyms), "Synonym pairs:", len(self.synonyms)

        list_of_simlex = []
        list_of_wordsim = []

        syn_batches = int(self.syn_count / self.batch_size)
        ant_batches = int(self.ant_count / self.batch_size)

        batches_per_epoch = syn_batches + ant_batches

        print "\nRunning the optimisation procedure for", self.max_iter, "iterations..."

        last_time = time.time()

        if self.log_scores_over_time:

            fwrite_simlex = open("results/simlex_scores.txt", "w")
            fwrite_wordsim = open("results/wordsim_scores.txt", "w")

        while current_iteration < self.max_iter:

            # how many attract/repel batches we've done in this epoch so far.
            antonym_counter = 0
            synonym_counter = 0

            order_of_synonyms = range(0, self.syn_count)
            order_of_antonyms = range(0, self.ant_count)

            random.shuffle(order_of_synonyms)
            random.shuffle(order_of_antonyms)

            # list of 0 where we run synonym batch, 1 where we run antonym batch
            list_of_batch_types = [0] * batches_per_epoch
            list_of_batch_types[syn_batches:] = [1] * ant_batches # all antonym batches to 1
            random.shuffle(list_of_batch_types)

            if current_iteration == 0:
                print "\nStarting epoch:", current_iteration+1, "\n"
            else:
                print "\nStarting epoch:", current_iteration+1, "Last epoch took:", round(time.time() - last_time, 1), "seconds. \n"
                last_time = time.time()


            for batch_index in range(0, batches_per_epoch):

                # we can Log SimLex / WordSim scores
                if self.log_scores_over_time and (batch_index % (batches_per_epoch/20) == 0):

                    (simlex_score, wordsim_score) = self.create_vector_dictionary()
                    list_of_simlex.append(simlex_score)
                    list_of_wordsim.append(wordsim_score)
                    
                    print >>fwrite_simlex,  len(list_of_simlex)+1, simlex_score
                    print >>fwrite_wordsim, len(list_of_simlex)+1, wordsim_score

                syn_or_ant_batch = list_of_batch_types[batch_index]

                if syn_or_ant_batch == 0:
                    # do one synonymy batch:

                    synonymy_examples = [self.synonyms[order_of_synonyms[x]] for x in range(synonym_counter * self.batch_size, (synonym_counter+1) * self.batch_size)]
                    current_negatives = self.extract_negative_examples(synonymy_examples, attract_batch=True)

                    self.sess.run([self.attract_cost_step], feed_dict={self.attract_examples: synonymy_examples, self.negative_examples_attract: current_negatives, \
                                                                  self.attract_margin: self.attract_margin_value, self.regularisation_constant: self.regularisation_constant_value})
                    synonym_counter += 1

                else:

                    antonymy_examples = [self.antonyms[order_of_antonyms[x]] for x in range(antonym_counter * self.batch_size, (antonym_counter+1) * self.batch_size)]
                    current_negatives = self.extract_negative_examples(antonymy_examples, attract_batch=False)

                    self.sess.run([self.repel_cost_step], feed_dict={self.repel_examples: antonymy_examples, self.negative_examples_repel: current_negatives, \
                                                                  self.repel_margin: self.repel_margin_value, self.regularisation_constant: self.regularisation_constant_value})

                    antonym_counter += 1

            current_iteration += 1
            self.create_vector_dictionary() # whether to print SimLex score at the end of each epoch


    def create_vector_dictionary(self):
        """
        Extracts the current word vectors from TensorFlow embeddings and (if print_simlex=True) prints their SimLex scores. 
        """
        log_time = time.time()

        [current_vectors] = self.sess.run([self.W_dynamic])
        self.word_vectors = {}
        for idx in range(0, self.vocabulary_size):
            self.word_vectors[self.inverted_index[idx]] = normalise_vector(current_vectors[idx, :])

        if self.log_scores_over_time or self.print_simlex:
            (score_simlex, score_wordsim) = simlex_scores(self.word_vectors, self.print_simlex)
            return (score_simlex, score_wordsim)

        return (1.0, 1.0)


def random_different_from(top_range, number_to_not_repeat):

    result = random.randint(0, top_range-1)
    while result == number_to_not_repeat:
        result = random.randint(0, top_range-1)

    return result


def mix_sampling(list_of_examples, negative_examples):
    """
    Converts half of the negative examples to random words from the batch (that are not in the given example pair).  
    """
    mixed_negative_examples = []
    batch_size = len(list_of_examples)

    for idx, (left_idx, right_idx) in enumerate(negative_examples):

        new_left = left_idx
        new_right = right_idx

        if random.random() >= 0.5:
            new_left = list_of_examples[random_different_from(batch_size, idx)][random.randint(0, 1)]
        
        if random.random() >= 0.5:
            new_right = list_of_examples[random_different_from(batch_size, idx)][random.randint(0, 1)]

        mixed_negative_examples.append((new_left, new_right))

    return mixed_negative_examples


def normalise_word_vectors(word_vectors, norm=1.0):
    """
    This method normalises the collection of word vectors provided in the word_vectors dictionary.
    """
    for word in word_vectors:
        word_vectors[word] /= math.sqrt((word_vectors[word]**2).sum() + 1e-6)
        word_vectors[word] = word_vectors[word] * norm
    return word_vectors


def load_word_vectors(file_destination):
    """
    This method loads the word vectors from the supplied file destination. 
    It loads the dictionary of word vectors and prints its size and the vector dimensionality. 
    """
    print "Loading pretrained word vectors from", file_destination
    word_dictionary = {}

    try:
        
        f = codecs.open(file_destination, 'r', 'utf-8') 

        for line in f:

            line = line.split(" ", 1)   
            key = unicode(line[0].lower())
            word_dictionary[key] = numpy.fromstring(line[1], dtype="float32", sep=" ")

    except:

        print "Word vectors could not be loaded from:", file_destination
        return {}

    print len(word_dictionary), "vectors loaded from", file_destination     

    return word_dictionary


def print_word_vectors(word_vectors, write_path):
    """
    This function prints the collection of word vectors to file, in a plain textual format. 
    """

    f_write = codecs.open(write_path, 'w', 'utf-8')

    for key in word_vectors:
        print >>f_write, key, " ".join(map(unicode, numpy.round(word_vectors[key], decimals=6))) 

    print "Printed", len(word_vectors), "word vectors to:", write_path


def simlex_analysis(word_vectors, language="english", source="simlex", add_prefixes=True):
    """
    This method computes the Spearman's rho correlation (with p-value) of the supplied word vectors. 
    """
    pair_list = []
    if source == "simlex":
        fread_simlex=codecs.open("evaluation/simlex-" + language + ".txt", 'r', 'utf-8')
    elif source == "simlex-old":
        fread_simlex=codecs.open("evaluation/simlex-english-old.txt", 'r', 'utf-8')
    elif source == "simverb":
        fread_simlex=codecs.open("evaluation/simverb.txt", 'r', 'utf-8')
    elif source == "wordsim":
        fread_simlex=codecs.open("evaluation/ws-353/wordsim353-" + language + ".txt", 'r', 'utf-8') # specify english, english-rel, etc.

    # needed for prefixes if we are adding these.
    lp_map = {}
    lp_map["english"] = u"en_"
    lp_map["german"] = u"de_"
    lp_map["italian"] = u"it_"
    lp_map["russian"] = u"ru_"
    lp_map["croatian"] = u"sh_"
    lp_map["hebrew"] = u"he_"

    line_number = 0
    for line in fread_simlex:

        if line_number > 0:

            tokens = line.split()
            word_i = tokens[0].lower()
            word_j = tokens[1].lower()
            score = float(tokens[2])

            if add_prefixes:
                word_i = lp_map[language] + word_i
                word_j = lp_map[language] + word_j

            if word_i in word_vectors and word_j in word_vectors:
                pair_list.append( ((word_i, word_j), score) )
            else:
                pass

        line_number += 1

    if not pair_list:
        return (0.0, 0)

    pair_list.sort(key=lambda x: - x[1])

    coverage = len(pair_list)

    extracted_list = []
    extracted_scores = {}

    for (x,y) in pair_list:

        (word_i, word_j) = x
        current_distance = distance(word_vectors[word_i], word_vectors[word_j]) 
        extracted_scores[(word_i, word_j)] = current_distance
        extracted_list.append(((word_i, word_j), current_distance))

    extracted_list.sort(key=lambda x: x[1])

    spearman_original_list = []
    spearman_target_list = []

    for position_1, (word_pair, score_1) in enumerate(pair_list):
        score_2 = extracted_scores[word_pair]
        position_2 = extracted_list.index((word_pair, score_2))
        spearman_original_list.append(position_1)
        spearman_target_list.append(position_2)

    spearman_rho = spearmanr(spearman_original_list, spearman_target_list)
    return round(spearman_rho[0], 3), coverage


def normalise_vector(v1):
    return v1 / norm(v1)


def distance(v1, v2, normalised_vectors=False):
    """
    Returns the cosine distance between two vectors. 
    If the vectors are normalised, there is no need for the denominator, which is always one. 
    """
    if normalised_vectors:
        return 1 - dot(v1, v2)
    else:
        return 1 - dot(v1, v2) / ( norm(v1) * norm(v2) )


def simlex_scores(word_vectors, print_simlex=True):

    for language in ["english", "german", "italian", "russian", "croatian", "hebrew"]:

        simlex_score, simlex_coverage = simlex_analysis(word_vectors, language)

        if language not in ["hebrew", "croatian"]:
            ws_score, ws_coverage = simlex_analysis(word_vectors, language, source="wordsim")
        else:
            ws_score = 0.0
            ws_coverage = 0

        if language == "english":
            simverb_score, simverb_coverage = simlex_analysis(word_vectors, language, source="simverb")

        if simlex_coverage > 0:

            if print_simlex:
    
                if language == "english":

                    simlex_old, cov_old = simlex_analysis(word_vectors, language, source="simlex-old")

                    print "SimLex score for", language, "is:", simlex_score, "Original SimLex score is:", simlex_old, "coverage:", simlex_coverage, "/ 999"
                    print "SimVerb score for", language, "is:", simverb_score, "coverage:", simverb_coverage, "/ 3500"
                    print "WordSim score for", language, "is:", ws_score, "coverage:", ws_coverage, "/ 353\n"

                elif language in ["italian", "german", "russian"]:
                    
                    print "SimLex score for", language, "is:", simlex_score, "coverage:", simlex_coverage, "/ 999"
                    print "WordSim score for", language, "is:", ws_score, "coverage:", ws_coverage, "/ 353\n"

                elif language in ["hebrew", "croatian"]:

                    print "SimLex score for", language, "is:", simlex_score, "coverage:", simlex_coverage, "/ 999\n"

        if language == "english":
            simlex_score_en = simlex_score
            ws_score_en = ws_score

    return simlex_score_en, ws_score_en


def run_experiment(config_filepath):
    """
    This method runs the counterfitting experiment, printing the SimLex-999 score of the initial
    vectors, then counter-fitting them using the supplied linguistic constraints. 
    We then print the SimLex-999 score of the final vectors, and save them to a .txt file in the 
    results directory.
    """
    current_experiment = ExperimentRun(config_filepath)
    
    current_experiment.attract_repel() 
    
    print "\nSimLex score (Spearman's rho coefficient) of the final vectors is:"
           
    simlex_scores(current_experiment.word_vectors), "\n"

    os.system("mkdir -p results")
    
    print_word_vectors(current_experiment.word_vectors, current_experiment.output_filepath)


def main():
    """
    The user can provide the location of the config file as an argument. 
    If no location is specified, the default config file (experiment_parameters.cfg) is used.
    """
    try:
        config_filepath = sys.argv[1]
    except:
        print "\nUsing the default config file: experiment_parameters.cfg\n"
        config_filepath = "experiment_parameters.cfg"

    run_experiment(config_filepath)


if __name__=='__main__':
    main()

