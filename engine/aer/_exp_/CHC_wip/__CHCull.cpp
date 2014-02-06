#include "CHCull.hpp"


// PUBLIC ___________________________________________  


CHCull::~CHCull()
{
  for (unsigned int i=0u; i<m_numNodes; ++i)
  {
    m_query[i]->release();
    delete m_query[i];
  }
}

void CHCull::init()
{
  /// Visibility status
  m_isVisible.resize(m_numNodes, true);
  
  // special case @frame 0 (so lastFrame = -1)
  m_lastVisited.resize(m_numNodes, -1);
  
  
  /// Create queries 
  m_query.resize(m_numNodes);
  for (unsigned int i=0u; i<m_numNodes; ++i)
  {
    m_query[i] = new aer::Query();
    m_query[i]->generate();
  }
}






//-- // // // // //

void CHCull::nastyNaiveApproch(const aer::Camera &camera, const unsigned int frameId)
{
  /// Nasty Naive Approch
  
  glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
  m_leavesToRender.clear();
    
  
  /// Setup the base view for the priority queue sorting
  m_pView = &camera;
  m_distanceQueue.setView3D( m_pView );
  
  /// 1) sort front to back
  for (int i=0; i<m_numNodes; ++i)
  {
    if (IsALeaf(m_tree[i])) {
      pushToDistanceQueue( i );
    }
  }
  
  while (!m_distanceQueue.empty())
  {
    Node *pNode = m_distanceQueue.pop();
  
  
    unsigned int offset = GetNodeOffset( *pNode );
    
    aer::Query* pQuery = m_query[ offset ];
    pQuery->begin( aer::Query::ANY_SAMPLES_PASSED );  
    
    if (m_isVisible[ pNode->id ]) 
    {
      //glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
      //glDepthMask( GL_TRUE );
      
      m_pMeshes[ offset ].draw();
      m_leavesToRender.insert( offset );
    } 
    else 
    {
      //glColorMask( GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
      //glDepthMask( GL_FALSE );
      
      m_attribs[ pNode->id ].aabb.render();
    }
    
    pQuery->end();
    
    m_queryQueue.push( QueryNode( pQuery, pNode) );
  }
  
  while (!m_queryQueue.empty())
  {
    aer::Query *Q = m_queryQueue.front().first;
    
    // Treats the query if available..
    if (Q->isResultAvailable())
    {
      QueryNode QN = m_queryQueue.front();
      m_queryQueue.pop();
      
      m_isVisible[ QN.second->id ] = (Q->getResultui() > 0);
    }
  }
}

//-- // // // // //








void CHCull::run(const aer::Camera &camera, const unsigned int frameId)
{

  /// Extract the view frustum plane
  aer::Vector4 frustumPlanes[aer::Frustum::kNumPlane];
  aer::Frustum::ExtractPlanes( camera.getViewProjectionMatrix(), true, frustumPlanes);//

  /// Reset the leavesToRender set
  m_leavesToRender.clear();
  
  
  /// Clear buffers for queries
  glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT ); // ~  
  
  //glColorMask( GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
  //glDepthMask( GL_FALSE );
  
  
  /// Setup the base view for the priority queue sorting
  m_pView = &camera;
  m_distanceQueue.setView3D( m_pView );
  
  /// Push the root
  pushToDistanceQueue( 0u );
  

  while (!m_distanceQueue.empty() || !m_queryQueue.empty())
  {
    /// 1 - Processed finished occlusion queries
    while (!m_queryQueue.empty())
    {
      aer::Query *Q = m_queryQueue.front().first;
      
      // Treats the query if available..
      if (Q->isResultAvailable())
      {
        QueryNode QN = m_queryQueue.front();
        m_queryQueue.pop();
        
        handleQueryResult( QN, frameId);
      }
      
      // .. otherwise performs a query on a previously visible node
      else if (!m_vqueue.empty())
      {
        issueQuery( m_vqueue.front(), true );
        m_vqueue.pop();
      }
    }
    
    
    /// 2 - Tree traversal
    if (!m_distanceQueue.empty())
    {
      Node *pNode = m_distanceQueue.pop();
      const unsigned int nodeId = pNode->id;
      
      // ERROR : was visible become always FALSE for the root (when going outside)
      if (isInsideViewFrustum( nodeId, frustumPlanes))
      {        
        bool bWasVisible = wasVisible( nodeId, frameId);
        
        /// Update node visited status
        m_isVisible[nodeId] = false;//
        m_lastVisited[nodeId] = frameId;//
        
        if (bWasVisible)
        {
          printf("was visible %u\n", nodeId);
          
          /// Enqueue previously visible nodes
          if (IsALeaf(*pNode) /*&& chcQueryReasonable(pNode)*/) {
            m_vqueue.push(pNode);
          }
          traverseNode(pNode);
        }
        else
        {
          printf("was invisible %u\n", nodeId);
          
          //
          queryPreviouslyInvisibleNode(pNode);
        }
      }
      else
      {
        printf("%u OUTSIDE frustum\n", nodeId);
        
        // Bad I think.. but useful too.. find out wtd !
        //m_isVisible[nodeId] = false; //test
      }
    }
    
    
    /// Issue remaining previously invisible node queries 
    /// (potentially per batch)
    /// TODO : proceed m_leavesToRender before
    if (m_distanceQueue.empty()) {
      issueMultiQueries();//
    }  
  } // END while


  /// Issue remaining previously visible node queries
  while (!m_vqueue.empty())
  {
    Node *pNode = m_vqueue.front();
    m_vqueue.pop();
    
    issueQuery( pNode, true );
  }
  
  
  glDepthMask( GL_TRUE );  
  glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

}



// PRIVATE ___________________________________________


void CHCull::traverseNode(Node *node)
{
  AER_ASSERT( node != NULL );

  unsigned int offset = GetNodeOffset( *node );
  
  if (aer::BVH::IsALeaf(*node))
  {
    // id of the leave to render
    m_leavesToRender.insert( offset );
    
    //m_pMeshes[ offset ].draw();//
  }
  else
  {
    pushToDistanceQueue( offset + 0u );
    pushToDistanceQueue( offset + 1u );
    //m_isVisible[ node->id ] = false;
  }
}

void CHCull::pullUpVisibility(Node *node)
{
  AER_ASSERT( node != NULL );
  
  while (!m_isVisible[ node->id ])
  {
    m_isVisible[ node->id ] = true;
    node = &m_tree[ node->parent ];
  }
}

void CHCull::handleQueryResult(QueryNode queryNode, const unsigned int frameId)
{
  aer::Query *Q = queryNode.first;
       Node  *N = queryNode.second;
         
  unsigned int numSamplesPassed = Q->getResultui();
  
  
  if (numSamplesPassed > kSamplesPassedTHRESHOLD)
  {
    // In this simple version MultiQueries are always inner nodes
    if (!IsALeaf(*N))
    {
      printf("MQ visible\n");
      //failed multiqueries (ie, Query on a InnerNode) 
      //=> perform individuals queries instead
      queryIndividualsNodes( N );
    }
    else
    {
      if (!wasVisible(N->id, frameId)) {
        traverseNode(N);
      }
      pullUpVisibility(N);
    }
  }
  else
  {
    m_isVisible[N->id] = false;
  }
}

void CHCull::queryPreviouslyInvisibleNode(Node *node)
{
  m_iqueue.push( node );
      
  if (m_iqueue.size() >= kInvisibleNodesBatchSize) {
    issueMultiQueries();
  }  
}


void CHCull::issueQuery(Node *node, bool bWasVisible)
{ 
  //printf("issue Query (%d) %u\n", bWasVisible, node->id);
   
  aer::Query *pQuery = m_query[node->id];
  aer::Query::Target target = (kSamplesPassedTHRESHOLD==0u) ? aer::Query::ANY_SAMPLES_PASSED
                                                            : aer::Query::SAMPLES_PASSED;
  
  pQuery->begin( target );

  if (bWasVisible) 
  { 
    // Render full objects  
    m_pMeshes[GetNodeOffset(*node)].draw();
  } 
  else 
  {
    // Render AABB
    m_attribs[node->id].aabb.render();
  }
  
  pQuery->end();
  
  
  m_queryQueue.push( QueryNode( pQuery, node) );
}

//
void CHCull::queryIndividualsNodes(Node *node)
{
  std::queue<Node*> toSearch;
  toSearch.push( node );
  
  while (!toSearch.empty())
  {
    Node *pNode = toSearch.front();
    toSearch.pop();
    
    if (IsALeaf(*pNode))
    {
      issueQuery( pNode );
      //m_iqueue.push( pNode );
    } 
    else 
    {
      unsigned int off = GetNodeOffset(*pNode);
      toSearch.push( &m_tree[off] );
      toSearch.push( &m_tree[off+1u] );
    }
  }
}

void CHCull::issueMultiQueries()
{
  
  while (!m_iqueue.empty())
  {
    Node *node = m_iqueue.front();
    m_iqueue.pop();
    
    
//---------
    std::queue<Node*> toSearch;
    aer::DistanceQueue<Node*> leaves(aer::FRONT_TO_BACK, m_pView); //
    
    toSearch.push( node );
    
    while (!toSearch.empty())
    {
      Node *pNode = toSearch.front(); 
      toSearch.pop();
      
      if (IsALeaf(*pNode)) 
      {
        leaves.push( pNode, m_attribs[pNode->id].aabb.getCenter() );
      } 
      else 
      {
        unsigned int off = GetNodeOffset(*pNode);
        toSearch.push( &m_tree[off] );
        toSearch.push( &m_tree[off+1u] );
      }
    }

    aer::Query::Target target = (kSamplesPassedTHRESHOLD==0u) ? aer::Query::ANY_SAMPLES_PASSED
                                                              : aer::Query::SAMPLES_PASSED;

    aer::Query *pQuery = m_query[node->id];
    pQuery->begin( target );

    while (!leaves.empty())
    {
      Node *N = leaves.pop();     

      //m_attribs[N->id].aabb.render();
      m_pMeshes[GetNodeOffset( *N )].draw();
    }
    
    pQuery->end();
    m_queryQueue.push( QueryNode( pQuery, node) );

//---------

  }
  
  /*
  while (!m_iqueue.empty())
  {
    MultiQuery *MQ = GetNextMultiQuery(m_iqueue);
    issueQuery(MQ);
    m_iqueue.popNodes(MQ);
  }
  */
}
