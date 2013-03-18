#based on the python version http://ebiquity.umbc.edu/blogger/2010/12/07/naive-bayes-classifier-in-50-lines/

class Label
  attr_reader :tag, :feature_name, :feature_value 
  def initialize(tag, feature_name, feature_value)
    @tag = tag
    @feature_name = feature_name
    @feature_value = feature_value
  end
  def eql?(other)
    self.tag == other.tag and self.feature_name == other.feature_name and self.feature_value == other.feature_value
  end
  def hash
    [@tag, @feature_name, @feature_value].hash
  end
end

class Model
  attr_accessor :training_file, :features, :feature_name_list, :feature_counts, :feature_vectors, :label_counts
  #attr_reader
  def initialize(arff_file)
    @training_file = arff_file
    @features = Hash.new
    @feature_name_list = Array.new
    @feature_counts = Hash.new #return 1 if no key
    @feature_vectors = Array.new
    @label_counts = Hash.new #return 0 if no key
  end
  
  def get_values
    File.open(@training_file, "r") do |infile|
      while (line = infile.gets)
        if line[0] != "@"
          @feature_vectors.push(line.strip.downcase.split(','))
        elsif not line.strip.downcase.include? '@data' and not line.downcase.start_with? '@relation'
          @feature_name_list.push(line.strip.split[1])
          @features[@feature_name_list.last] = line.slice(line.index('{')+1..line.index('}')-1).strip.split(',')
        end
      end
    end
  end
  
  def train_classifier
    @feature_vectors.each do |fv|
      if @label_counts.has_key?(fv.last)
        @label_counts[fv.last] += 1
      else
        @label_counts[fv.last] = 1
      end
      (0..fv.length-1).each do |counter|
        label = Label.new(fv.last, @feature_name_list[counter], fv[counter])
        if @feature_counts.has_key?(label)
          @feature_counts[label] += 1
        else
          @feature_counts[label] = 1
        end
      end
    end
  end
  
  def classify(feature_vector)
    probability_per_label = Hash.new
    @label_counts.keys.each do |label|
      log_prob = 0
      feature_vector.each do |feature_value|
        log_prob += Math.log(@feature_counts[Label.new(label, @feature_name_list[feature_vector.index(feature_value)], feature_value)].to_f / @label_counts[label].to_f)
      end 
      probability_per_label[label] = (@label_counts[label].to_f/@label_counts.values.inject{|sum,x| sum + x }).to_f * Math.exp(log_prob).to_f
    end
    probability_per_label.max_by{|k,v| v}[0]
  end
  
  def test_classifier(arff_file)
    File.open(arff_file, "r") do |infile|
      while (line = infile.gets)
        if line[0] != "@"
          vector = line.strip.downcase.split(',')
          puts "classifier: " + classify(vector) + " given " + vector.last    
        end
      end
    end 
  end
end

m=Model.new("tennis.arff")
m.get_values
m.train_classifier
m.test_classifier("tennis.arff")