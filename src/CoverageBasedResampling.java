/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package weka.classifiers.meta;

import weka.classifiers.Classifier;
import weka.classifiers.RandomizableParallelIteratedSingleClassifierEnhancer;
import weka.core.*;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

import java.util.*;

/**
 <!-- globalinfo-start -->
 * Class for coverage balanced resampling meta classifier. Can do classification and regression depending on the base learner. <br/>
 * <br/>
 * this class is based on the paper "Coverage-based resampling: Building robust consolidated decision trees." Knowledge-Based Systems 79 (2015): 51-67.
 * <br/>
 * By Ibarguren, Igor, Jesús M. Pérez, Javier Muguerza, Ibai Gurrutxaga, and Olatz Arbelaitz.
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{Ibarguren2015,
 *    author = {Ibarguren Igor},
 *    journal = {Knowledge-Based Systems},
 *    pages = {51-67},
 *    title = {Coverage-based resampling: Building robust consolidated decision trees},
 *    volume = {79},
 *    year = {2015}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 *
 * <pre> -C
 *  the expected coverage in percentage of the majority class
 *  training set size. (default 90, cannot be 100)</pre>
 *
 * <pre> -useCoverage
 *  a boolean represents if we use fixed size of iterations (false) or coverage based (true).</pre>
 *
 * <pre> -S &lt;num&gt;
 *  Random number seed.
 *  (default 1)</pre>
 *
 * <pre> -num-slots &lt;num&gt;
 *  Number of execution slots.
 *  (default 1 - i.e. no parallelism)</pre>
 *
 * <pre> -I &lt;num&gt;
 *  Number of iterations.
 *  (default 10)</pre>
 *
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 *
 * <pre> -W
 *  Full name of base classifier.
 *  (default: weka.classifiers.trees.REPTree)</pre>
 *
 * <pre> 
 * Options specific to classifier weka.classifiers.trees.REPTree:
 * </pre>
 *
 * <pre> -M &lt;minimum number of instances&gt;
 *  Set minimum number of instances per leaf (default 2).</pre>
 *
 * <pre> -V &lt;minimum variance for split&gt;
 *  Set minimum numeric class variance proportion
 *  of train variance for split (default 1e-3).</pre>
 *
 * <pre> -N &lt;number of folds&gt;
 *  Number of folds for reduced error pruning (default 3).</pre>
 *
 * <pre> -S &lt;seed&gt;
 *  Seed for random data shuffling (default 1).</pre>
 *
 * <pre> -P
 *  No pruning.</pre>
 *
 * <pre> -L
 *  Maximum tree depth (default -1, no maximum)</pre>
 *
 * <pre> -I
 *  Initial class value count (default 0)</pre>
 *
 * <pre> -R
 *  Spread initial count over all class values (i.e. don't use 1 per value)</pre>
 *
 <!-- options-end -->
 *
 * Options after -- are passed to the designated classifier.<p>
 *
 * @author Itay Hazan (itayhaz@post.bgu.ac.il)
 * @author Andrey Finkelstein (andreyfi@post.bgu.ac.il)
 * @version $Revision: 1 $
 * this implementation is done by converting the Bagging classifier implemented by :
 * Eibe Frank (eibe@cs.waikato.ac.nz), Len Trigg (len@reeltwo.com), Richard Kirkby (rkirkby@cs.waikato.ac.nz)
*/
public class CoverageBasedResampling
        extends RandomizableParallelIteratedSingleClassifierEnhancer
        implements WeightedInstancesHandler, TechnicalInformationHandler, PartitionGenerator, Aggregateable<CoverageBasedResampling> {

    /** the coverage percentage of the majority class */
    protected double m_MajorityCoverage = 90;

    /** if using coverage(false) or a fixed number of iterations(true)*/
    protected boolean m_UseNumIterations = false;


    /** splitted instances by the classes*/
    protected List<List<Instance>> m_splittedInstances;


    public String majorityCoverageTipText() {
        return "The majority class coverage you'd like to pursue.";
    }

    public String useNumIterationTipText() {
        return "whether to use fixed number of iterations instead of fixed coverage";
    }
    public String seedTipText() {
        return "The seed used for randomizing the data ";
    }

    /**
     * Constructor.
     */
    public CoverageBasedResampling() {
        m_Classifier = new weka.classifiers.trees.REPTree();
    }



    /**
     * Returns a string describing classifier
     * @return a description suitable for
     * displaying in the explorer/experimenter gui
     */
    public String globalInfo() {

        return "Class for coverage resampling a meta-classifier to use only for classification "
                + "For more information, see\n\n"
                + getTechnicalInformation().toString();
    }

    /**
     * Returns an instance of a TechnicalInformation object, containing
     * detailed information about the technical background of this class,
     * e.g., paper reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    @Override
    public TechnicalInformation getTechnicalInformation() {

        TechnicalInformation result = new TechnicalInformation(Type.ARTICLE);
        result.setValue(Field.AUTHOR, "Ibarguren Igor");
        result.setValue(Field.YEAR, "2015");
        result.setValue(Field.TITLE, "Coverage-based resampling");
        result.setValue(Field.JOURNAL, "Knowledge-Based Systems");
        result.setValue(Field.VOLUME, "79");
        result.setValue(Field.PAGES, "51-67");

        return result;
    }

    /**
     * String describing default classifier.
     * @return the default classifier classname
     */
    @Override
    protected String defaultClassifierString() {
        return "weka.classifiers.trees.REPTree";
    }

    /**
     * Returns an enumeration describing the available options.
     * @return an enumeration of all the available options.
     */
    @Override
    public Enumeration<Option> listOptions() {

        Vector<Option> newVector = new Vector<Option>(3);

        newVector.addElement(new Option(
                "\tExpected value of the coverage on the majority class in percentage\n",
                "C", 90, "-C"));
        newVector.addElement(new Option(
                "\tuse coverage instead of fixed size.",
                "useNumIterations", 0, "-useNumIterations"));

        newVector.addAll(Collections.list(super.listOptions()));

        return newVector.elements();
    }


    /**
     * Parses a given list of options. <p/>
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    @Override
    public void setOptions(String[] options) throws Exception {

        String coverage = Utils.getOption('C', options);
        if (coverage.length() != 0) {
            setMajorityCoverage(Integer.parseInt(coverage));
        } else {
            setMajorityCoverage(90);
        }

        setM_UseNumIterations(Utils.getFlag("-useNumIterations", options));

        super.setOptions(options);
        Utils.checkForRemainingOptions(options);
    }

    /**
     * Gets the current settings of the Classifier.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    @Override
    public String [] getOptions() {

        Vector<String> options = new Vector<String>();

        options.add("-C");
        options.add("" + getMajorityCoverage());
        if (isM_UseNumIterations())
            options.add("-useNumIterations");


        Collections.addAll(options, super.getOptions());

        return options.toArray(new String[0]);
    }

    public boolean isM_UseNumIterations() {
        return m_UseNumIterations;
    }

    public void setM_UseNumIterations(boolean m_UseNumIterations) {
        this.m_UseNumIterations = m_UseNumIterations;
    }

    public double getMajorityCoverage() {
        return m_MajorityCoverage;
    }

    public void setMajorityCoverage(double m_CoverageClass) {
        this.m_MajorityCoverage = m_CoverageClass;
    }


    protected Random m_random;
    protected Instances m_data;

    /**
     * Returns a training set for a particular iteration.
     * this is the heart of the method. it balanced all the data according to the minority class
     *
     * @param iteration the number of the iteration for the requested training set.
     * @return a balanced training set for the supplied iteration number
     * @throws Exception if something goes wrong when generating a training set.
     */
    @Override
    protected synchronized Instances getTrainingSet(int iteration) throws Exception {

        Instances toReturn = new Instances(m_data, m_data.size());
        AttributeStats classAttributeStats = m_data.attributeStats(m_data.classIndex());
        double[] nominalWeights = classAttributeStats.nominalWeights;
        int minorityIndex = Utils.minIndex(nominalWeights);
        double minoritySize = nominalWeights[minorityIndex];
        Random r = new Random(m_Seed + iteration);
        for (int i = 0; i < m_data.numClasses(); i++) {
            List<Instance> currentClass = m_splittedInstances.get(i);
            if(i==minorityIndex){
                for (int j = 0; j < currentClass.size(); j++) {
                    toReturn.add((Instance) currentClass.get(j).copy());
                }
            }
            else{
                Set<Integer> selectedSet = new HashSet<Integer>();
                while (selectedSet.size() < minoritySize) {
                    selectedSet.add(r.nextInt((int) nominalWeights[i]));
                }
                for (int index : selectedSet)
                    toReturn.add(m_splittedInstances.get(i).get(index));
            }

        }
        return toReturn;
    }

    /**
     * @param data the training data to be used for generating the sub samples for the inner classifier.
     * @throws Exception if the classifier could not be built successfully
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {

        if (data.classAttribute().isNumeric()){
            throw new Exception("Numeric Class Attribute is not supported");
        }

        m_splittedInstances = new ArrayList<List<Instance>>();
        for (int i = 0; i < data.numClasses(); i++) {
            List<Instance> current = new LinkedList<Instance>();
            m_splittedInstances.add(i,current);
        }

        Enumeration<Instance> instanceEnumeration = data.enumerateInstances();
        while (instanceEnumeration.hasMoreElements()){
            Instance currentInstance = instanceEnumeration.nextElement();
            m_splittedInstances.get((int)currentInstance.classValue()).add(currentInstance);
        }

        // check if the fixed sample size argument is legal
        if (m_UseNumIterations)
            m_NumIterations = getNumIterations();
        else {
            if (m_MajorityCoverage >= 100 || m_MajorityCoverage <=0) {
                throw new IllegalArgumentException("The coverage percentage must be " +
                        "larger than 0 and lower than 100 Received " + m_MajorityCoverage);
            }
            else{
                AttributeStats classAttributeStats = data.attributeStats(data.classIndex());
                double minorityInstancesNum = classAttributeStats.nominalWeights[Utils.minIndex(classAttributeStats.nominalWeights)];
                double majorityInstancesNum = classAttributeStats.nominalWeights[Utils.maxIndex(classAttributeStats.nominalWeights)];
                double minorityMajorityShare = minorityInstancesNum/majorityInstancesNum;
                double calculatedIterations = Math.log(1 - m_MajorityCoverage / 100.0) / Math.log(1 - minorityMajorityShare);
                int iterationsNeeded = (int) Math.ceil(calculatedIterations);
                m_NumIterations = iterationsNeeded;
            }
        }
        // can classifier handle the data?
        getCapabilities().testWithFail(data);

        // get fresh Instances object
        m_data = new Instances(data);

        super.buildClassifier(m_data);

        m_random = new Random(m_Seed);

        for (int j = 0; j < m_Classifiers.length; j++) {
            if (m_Classifier instanceof Randomizable) {
                ((Randomizable) m_Classifiers[j]).setSeed(m_random.nextInt());
            }
        }

        buildClassifiers();
        m_data = null;
    }


    /**
     * Calculates the class membership probabilities for the given test
     * instance.
     *
     * @param instance the instance to be classified
     * @return preedicted class probability distribution
     * @throws Exception if distribution can't be computed successfully
     */
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {

        double [] sums = new double [instance.numClasses()], newProbs;

        if (instance.classAttribute().isNumeric()){
            throw new Exception("Numeric Class Attribute is not supported");
        }

        for (int i = 0; i < m_NumIterations; i++) {
            newProbs = m_Classifiers[i].distributionForInstance(instance);
            for (int j = 0; j < newProbs.length; j++)
                sums[j] += newProbs[j];
        }
        if (Utils.eq(Utils.sum(sums), 0)) {
            return sums;
        } else {
            Utils.normalize(sums);
            return sums;
        }
    }

    /**
     * @return description of the subsampled classifier as a string
     */
    @Override
    public String toString() {

        if (m_Classifiers == null) {
            return "No model built yet.";
        }
        StringBuffer text = new StringBuffer();
        text.append("All the base classifiers: \n\n");
        for (int i = 0; i < m_Classifiers.length; i++)
            text.append(m_Classifiers[i].toString() + "\n\n");

        return text.toString();
    }

    /**
     * Builds the classifier to generate a partition.
     */
    @Override
    public void generatePartition(Instances data) throws Exception {

        if (m_Classifier instanceof PartitionGenerator)
            buildClassifier(data);
        else throw new Exception("Classifier: " + getClassifierSpec()
                + " cannot generate a partition");
    }

    /**
     * Computes an array that indicates leaf membership
     */
    @Override
    public double[] getMembershipValues(Instance inst) throws Exception {

        if (m_Classifier instanceof PartitionGenerator) {
            ArrayList<double[]> al = new ArrayList<double[]>();
            int size = 0;
            for (int i = 0; i < m_Classifiers.length; i++) {
                double[] r = ((PartitionGenerator)m_Classifiers[i]).
                        getMembershipValues(inst);
                size += r.length;
                al.add(r);
            }
            double[] values = new double[size];
            int pos = 0;
            for (double[] v: al) {
                System.arraycopy(v, 0, values, pos, v.length);
                pos += v.length;
            }
            return values;
        } else throw new Exception("Classifier: " + getClassifierSpec()
                + " cannot generate a partition");
    }

    /**
     * Returns the number of elements in the partition.
     */
    @Override
    public int numElements() throws Exception {

        if (m_Classifier instanceof PartitionGenerator) {
            int size = 0;
            for (int i = 0; i < m_Classifiers.length; i++) {
                size += ((PartitionGenerator)m_Classifiers[i]).numElements();
            }
            return size;
        } else throw new Exception("Classifier: " + getClassifierSpec()
                + " cannot generate a partition");
    }

    /**
     * Returns the revision string.
     *
     * @return		the revision
     */
    @Override
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 1 $");
    }

    /**
     * Main method for testing this class.
     *
     * @param argv the options
     */
    public static void main(String [] argv) {
        runClassifier(new CoverageBasedResampling(), argv);
    }

    protected List<Classifier> m_classifiersCache;

    /**
     * Aggregate an object with this one
     *
     * @param toAggregate the object to aggregate
     * @return the result of aggregation
     * @throws Exception if the supplied object can't be aggregated for some
     *           reason
     */
    @Override
    public CoverageBasedResampling aggregate(CoverageBasedResampling toAggregate) throws Exception {
        if (!m_Classifier.getClass().isAssignableFrom(toAggregate.m_Classifier.getClass())) {
            throw new Exception("Can't aggregate because base classifiers differ");
        }

        if (m_classifiersCache == null) {
            m_classifiersCache = new ArrayList<Classifier>();
            m_classifiersCache.addAll(Arrays.asList(m_Classifiers));
        }
        m_classifiersCache.addAll(Arrays.asList(toAggregate.m_Classifiers));

        return this;
    }

    /**
     * Call to complete the aggregation process. Allows implementers to do any
     * final processing based on how many objects were aggregated.
     *
     * @throws Exception if the aggregation can't be finalized for some reason
     */
    @Override
    public void finalizeAggregation() throws Exception {
        m_Classifiers = m_classifiersCache.toArray(new Classifier[1]);
        m_NumIterations = m_Classifiers.length;

        m_classifiersCache = null;
    }
}

