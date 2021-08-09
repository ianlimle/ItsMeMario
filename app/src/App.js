import React, { useState } from 'react';
import './App.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import { Container, Row, Col, Form, Button } from 'react-bootstrap';

function App() {
  const [trainingParameters, setTrainingParameters] = useState({
    episodes: '500',
    memory: '1000',
    explorationDecay: '0.999975',
    syncingStep: '40',
    learningStep: '40',
  });
  const [filePath, setFilePath] = useState('');
  const [message, setMessage] = useState('');

  const requestOptions_POST = {
    method: 'POST',
    mode: 'cors',
    headers: { 'Content-Type': 'application/json' },
  };

  const onTrain = async (e) => {
    e.preventDefault();
    try {
      await fetch(`http://localhost:5001/train`, {
        ...requestOptions_POST,
        body: JSON.stringify({
          memory: trainingParameters.memory,
          exploration_rate_decay: trainingParameters.explorationDecay,
          learn_every: trainingParameters.learningStep,
          sync_every: trainingParameters.syncingStep,
          episodes: trainingParameters.episodes,
        }),
      });
    } catch (error) {
      setMessage('Something went wrong');
    }
  };

  const onReplay = async (e) => {
    e.preventDefault();
    try {
      await fetch(`http://localhost:5001/test`, {
        ...requestOptions_POST,
        body: JSON.stringify({
          checkpoint: filePath,
        }),
      });
    } catch (error) {
      setMessage('Something went wrong');
    }
  };

  const onPlay = async (e) => {
    e.preventDefault();
    try {
      await fetch(`http://localhost:5001/play`, requestOptions_POST);
    } catch (error) {
      setMessage('Something went wrong');
    }
  };

  const handleChange = (e) => {
    setTrainingParameters({
      ...trainingParameters,
      [e.target.name]: e.target.value,
    });
  };

  return (
    <div className='App background-styling'>
      <h1>ITS ME MARIOOOOO</h1>
      <p>Ong Xiang Qian | Mong Chang Hsi | Chok Hao Ze | Ian Lim</p>
      <Container>
        <Row>
          <Col>
            <h2>Train!</h2>
            <p>
              Fill in the following training parameters to start training your
              own Mario!
            </p>

            <Form onSubmit={(e) => onTrain(e)}>
              <Form.Group as={Row} className='mb-3' controlId='formEpisodes'>
                <Form.Label column sm={6}>
                  Episodes
                </Form.Label>
                <Col sm={6}>
                  <Form.Control
                    name='episodes'
                    type='string'
                    placeholder='Episodes'
                    value={trainingParameters.episodes}
                    onChange={(e) => handleChange(e)}
                  />
                </Col>
              </Form.Group>

              <Form.Group as={Row} className='mb-3' controlId='formMemory'>
                <Form.Label column sm={6}>
                  Memory
                </Form.Label>
                <Col sm={6}>
                  <Form.Control
                    name='memory'
                    type='string'
                    placeholder='Memory'
                    value={trainingParameters.memory}
                    onChange={(e) => handleChange(e)}
                  />
                </Col>
              </Form.Group>
              <Form.Group
                as={Row}
                className='mb-3'
                controlId='formExplorationDecayRate'
              >
                <Form.Label column sm={6}>
                  Exploration Decay Rate
                </Form.Label>
                <Col sm={6}>
                  <Form.Control
                    name='explorationDecay'
                    type='string'
                    placeholder='Exploration Decay Rate'
                    value={trainingParameters.explorationDecay}
                    onChange={(e) => handleChange(e)}
                  />
                </Col>
              </Form.Group>
              <Form.Group as={Row} className='mb-3' controlId='formSyncingStep'>
                <Form.Label column sm={6}>
                  Syncing Step
                </Form.Label>
                <Col sm={6}>
                  <Form.Control
                    name='syncingStep'
                    type='string'
                    placeholder='Syncing Step'
                    value={trainingParameters.syncingStep}
                    onChange={(e) => handleChange(e)}
                  />
                </Col>
              </Form.Group>
              <Form.Group
                as={Row}
                className='mb-3'
                controlId='formLearningStep'
              >
                <Form.Label column sm={6}>
                  Learning Step
                </Form.Label>
                <Col sm={6}>
                  <Form.Control
                    name='learningStep'
                    type='string'
                    placeholder='Learning Step'
                    value={trainingParameters.learningStep}
                    onChange={(e) => handleChange(e)}
                  />
                </Col>
              </Form.Group>

              <Button variant='primary' type='submit'>
                Train
              </Button>
            </Form>
          </Col>
          <Col>
            <h2>Replay!</h2>
            <p>
              Upload one of your trained checkpoints to see your Mario in
              action!
            </p>
            <Form onSubmit={(e) => onReplay(e)}>
              <Form.Group controlId='formFile' className='mb-3'>
                <Form.Label>Checkpoint File Path</Form.Label>
                <Form.Control
                  name='filePath'
                  type='string'
                  placeholder='File Path'
                  value={filePath}
                  onChange={(e) => setFilePath(e.target.value)}
                />
              </Form.Group>
              <Button variant='primary' type='submit'>
                Replay
              </Button>
            </Form>
          </Col>
          <Col>
            <h2>Play!</h2>
            <p>
              Bored while waiting for your Mario to be trained? Why not play one
              game yourself!
            </p>
            <Button onClick={(e) => onPlay(e)}>Play</Button>
          </Col>
        </Row>
      </Container>
      {message && <p>{message}</p>}
    </div>
  );
}

export default App;
