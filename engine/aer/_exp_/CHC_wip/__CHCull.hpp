#ifndef CHCULL_HPP_
#define CHCULL_HPP_

#include <iostream> // for std::pair
#include <queue>
#include <set>

#include <aerBVH.hpp>
#include <aerDistanceQueue.hpp>
#include <view/aerCamera.hpp>
#include <device/aerQuery.hpp>



class CHCull : public aer::BVH
{
  // ================
  // ++ STRUCTURES ++
  // ================
  public:
    typedef std::set<unsigned int>      VLeavesId_t;
    typedef VLeavesId_t::const_iterator VLeavesIdCIterator_t;
    
  private:
    typedef std::pair<aer::Query*, Node*> QueryNode;
    

  // ===================
  // ++ STATIC FIELDS ++
  // ===================
  private:
    static const unsigned int kSamplesPassedTHRESHOLD  = 0u;  //
    static const unsigned int kInvisibleNodesBatchSize = 20u; //
  
  
  // ============
  // ++ FIELDS ++
  // ============
  private:
    /// Nodes attribs (~)
    std::vector<bool>         m_isVisible;
    std::vector<unsigned int> m_lastVisited; //
    std::vector<aer::Query*>  m_query;
    
    
    /// Queue used for the front-to-back ordering of the processed nodes
    aer::DistanceQueue<Node*> m_distanceQueue;
    const aer::View3D *m_pView;// ~ XXX
    
    /// Previously visible and invisible nodes to be queried
    std::queue<Node*> m_vqueue;
    std::queue<Node*> m_iqueue;
    
    /// Proceeding queries with their associated nodes
    std::queue<QueryNode> m_queryQueue;
    
    /// Indices of the visible leaves to render
    std::set<unsigned int> m_leavesToRender;
      
  

  // =============
  // ++ METHODS ++
  // =============
  public:
    CHCull() 
      : m_distanceQueue(aer::FRONT_TO_BACK) 
    {}
    
    ~CHCull();
    
    
    void run(const aer::Camera &camera, const unsigned int frameId);
  
    void nastyNaiveApproch(const aer::Camera &camera, const unsigned int frameId);
    
  
    /// Return the set of visible leaves
    const VLeavesId_t& getVisibleLeavesId() const {return m_leavesToRender;} // ~
  
  
  private:
    virtual void init();
    
    
    void traverseNode(Node *node);
    
    /// Set the node and its ancestors to be visible
    void pullUpVisibility(Node *node);
    
    void handleQueryResult(QueryNode queryNode, const unsigned int frameId);
    
    void queryPreviouslyInvisibleNode(Node *node);
    
    void issueQuery(Node *node, bool bWasVisible=false);
    
    void queryIndividualsNodes(Node *node);
    
    void issueMultiQueries();
    
    
    //--------------------------
    
    
    /// Return true if the node was previously visible
    /// Tests the node visibility and previous frame visit, therefore
    /// it is valid only before any update of the node attributes.
    bool wasVisible(const unsigned int nodeId, const unsigned int frameId)
    {
      return m_isVisible[nodeId] && (m_lastVisited[nodeId] == frameId-1u);
    }
    
    /// Push a node in the priority queue based on its node ID
    void pushToDistanceQueue(const unsigned int nodeId)
    {
      m_distanceQueue.push( &m_tree[nodeId], m_attribs[nodeId].aabb.getCenter());
    }
};

#endif // CHCULL_HPP_
