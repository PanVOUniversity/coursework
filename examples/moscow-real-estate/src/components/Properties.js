import React from 'react';
import { motion } from 'framer-motion';
import { FaBed, FaBath, FaCar, FaMapMarkerAlt } from 'react-icons/fa';
import './Properties.css';

const Properties = () => {
  const properties = [
    {
      id: 1,
      title: 'Пентхаус в Москва-Сити',
      location: 'Москва-Сити',
      price: '₽150,000,000',
      type: 'Продажа',
      bedrooms: 4,
      bathrooms: 3,
      parking: 2,
      image: 'https://images.unsplash.com/photo-1545324418-cc1a3fa10c00?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80'
    },
    {
      id: 2,
      title: 'Апартаменты на Арбате',
      location: 'Арбат, Москва',
      price: '₽120,000/месяц',
      type: 'Аренда',
      bedrooms: 2,
      bathrooms: 1,
      parking: 1,
      image: 'https://images.unsplash.com/photo-1564013799919-ab600027ffc6?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80'
    },
    {
      id: 3,
      title: 'Квартира в центре',
      location: 'Центр Москвы',
      price: '₽80,000,000',
      type: 'Продажа',
      bedrooms: 3,
      bathrooms: 2,
      parking: 1,
      image: 'https://images.unsplash.com/photo-1600607687939-ce8a6c25118c?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80'
    }
  ];

  return (
    <section id="properties" className="properties">
      <div className="container">
        <motion.div
          className="section-header"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <h2 className="section-title">Премиальные объекты</h2>
          <p className="section-subtitle">
            Эксклюзивная подборка лучших объектов недвижимости в Москве
          </p>
        </motion.div>

        <div className="properties-grid">
          {properties.map((property, index) => (
            <motion.div
              key={property.id}
              className="property-card"
              initial={{ opacity: 0, y: 50 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.2 }}
              viewport={{ once: true }}
              whileHover={{ y: -10 }}
            >
              <div className="property-image-container">
                <img
                  src={property.image}
                  alt={property.title}
                  className="property-image"
                />
                <div className="property-badge">{property.type}</div>
              </div>

              <div className="property-content">
                <h3 className="property-title">{property.title}</h3>
                <p className="property-location">
                  <FaMapMarkerAlt />
                  {property.location}
                </p>

                <div className="property-details">
                  <div className="detail-item">
                    <FaBed />
                    <span>{property.bedrooms} спален</span>
                  </div>
                  <div className="detail-item">
                    <FaBath />
                    <span>{property.bathrooms} ванных</span>
                  </div>
                  <div className="detail-item">
                    <FaCar />
                    <span>{property.parking} гараж</span>
                  </div>
                </div>

                <div className="property-footer">
                  <div className="property-price">{property.price}</div>
                  <motion.button
                    className="btn btn-primary btn-sm"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    Подробнее
                  </motion.button>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Properties;
